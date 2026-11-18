from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

FEATURES = ['distance_m', 'distance_rank', 'heading_cos', 'heading_sin_abs',
            'segment_length_m', 'projection_fraction', 'speed_mps', 'dt_s',
            'previous_distance_m', 'next_distance_m', 'start_degree', 'end_degree']


@dataclass
class Trip:
    trip_id: str
    nodes: np.ndarray
    arcs: np.ndarray
    track: np.ndarray
    original_rows: np.ndarray
    xy: np.ndarray
    gps: np.ndarray
    lengths: np.ndarray
    graph: csr_matrix
    issues: dict


def load_trip(folder):
    """Load inference inputs only. This function never opens a .route file."""
    folder = Path(folder)
    def read(ext, columns, dtype=float):
        value = np.loadtxt(folder / (folder.name + ext), dtype=dtype, ndmin=2)
        if value.shape[1] != columns or not len(value):
            raise ValueError(f'{ext}: expected {columns} columns and nonempty data')
        return value
    nodes, arcs, raw = read('.nodes', 2), read('.arcs', 2, int), read('.track', 3)
    if not np.isfinite(nodes).all() or np.any(np.abs(nodes) > [180, 90]):
        raise ValueError('Invalid network coordinates')
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError('Arc endpoint outside node array')
    valid = np.isfinite(raw).all(axis=1) & (np.abs(raw[:, :2]) <= [180, 90]).all(axis=1)
    rows = np.flatnonzero(valid)
    # Stable chronological order; duplicate timestamps retained, dt guarded in features.
    rows = rows[np.argsort(raw[rows, 2], kind='stable')]
    track = raw[rows]
    if not len(track):
        raise ValueError('Trip has no valid GPS observations')
    origin = nodes.mean(axis=0)
    scale = np.array([np.cos(np.deg2rad(origin[1])), 1.0]) * (np.pi / 180) * 6371008.8
    xy, gps = (nodes - origin) * scale, (track[:, :2] - origin) * scale
    lengths = np.linalg.norm(xy[arcs[:, 1]] - xy[arcs[:, 0]], axis=1)
    # Collapse duplicate directed edges with minimum length (not sparse summation).
    edges = {}
    for (u, v), w in zip(arcs, lengths):
        edges[(int(u), int(v))] = min(edges.get((int(u), int(v)), np.inf), max(float(w), 1e-6))
    graph = csr_matrix(([v for v in edges.values()],
                        ([k[0] for k in edges], [k[1] for k in edges])), shape=(len(nodes), len(nodes)))
    dt = np.diff(track[:, 2])
    issues = {'invalid_observations': int((~valid).sum()),
              'out_of_order_steps': int((np.diff(raw[valid, 2]) < 0).sum()),
              'duplicate_timestamps': int((dt == 0).sum()),
              'gaps_over_10s': int((dt > 10).sum()),
              'max_gap_s': float(dt.max()) if len(dt) else 0.0,
              'weak_components': int(connected_components(graph, directed=True, connection='weak')[0]),
              'isolated_nodes': int(((np.bincount(arcs.ravel(), minlength=len(nodes))) == 0).sum()),
              'zero_length_arcs': int((lengths < 1e-6).sum()),
              'raw_observations': len(raw)}
    return Trip(folder.name, nodes, arcs, track, rows, xy, gps, lengths, graph, issues)


def project(points, starts, ends):
    d = ends - starts
    frac = np.sum((points - starts) * d, axis=-1) / np.maximum(np.sum(d*d, axis=-1), 1e-12)
    frac = np.clip(frac, 0, 1)
    snapped = starts + frac[..., None] * d
    return np.linalg.norm(points - snapped, axis=-1), frac


def candidates(trip, k=12):
    """Sample network segments every <=30m, shortlist, then exact segment projection.

    KD-tree retrieval is approximate for dense networks; candidate miss rate is measured.
    Both directed arcs remain separate. All feature inputs are label-free.
    """
    k = min(k, len(trip.arcs))
    starts, ends = trip.xy[trip.arcs[:, 0]], trip.xy[trip.arcs[:, 1]]
    counts = np.maximum(2, np.ceil(trip.lengths / 30).astype(int) + 1)
    sample_arcs = np.repeat(np.arange(len(counts)), counts)
    fractions = np.concatenate([np.linspace(0, 1, n) for n in counts])
    samples = starts[sample_arcs] + fractions[:, None] * (ends-starts)[sample_arcs]
    tree = cKDTree(samples)
    _, nearest = tree.query(trip.gps, k=min(len(samples), max(96, k * 12)))
    nearest = np.asarray(nearest).reshape(len(trip.gps), -1)
    ids, distances, fractions_out = [], [], []
    for i, hit in enumerate(nearest):
        arc_ids = np.unique(sample_arcs[hit])
        if len(arc_ids) < k:
            _, more = tree.query(trip.gps[i], k=min(len(samples), 768))
            arc_ids = np.unique(sample_arcs[np.atleast_1d(more)])
        if len(arc_ids) < k:
            arc_ids = np.arange(len(trip.arcs))
        dist, frac = project(trip.gps[i], starts[arc_ids], ends[arc_ids])
        order = np.lexsort((arc_ids, dist))[:k]
        ids.append(arc_ids[order]); distances.append(dist[order]); fractions_out.append(frac[order])
    ids, distances, fractions_out = np.array(ids), np.array(distances), np.array(fractions_out)
    n = len(trip.gps)
    prev = np.r_[0, np.arange(n-1)]
    nxt = np.r_[np.arange(1, n), n-1]
    motion = trip.gps[nxt] - trip.gps[prev]
    norm = np.linalg.norm(motion, axis=1)
    motion /= np.maximum(norm[:, None], 1e-9)
    directions = ends[ids] - starts[ids]
    directions /= np.maximum(trip.lengths[ids, None], 1e-9)
    cosine = np.sum(directions * motion[:, None, :], axis=-1)
    sine = np.abs(directions[..., 0]*motion[:, None, 1] - directions[..., 1]*motion[:, None, 0])
    dt = np.maximum(trip.track[nxt, 2]-trip.track[prev, 2], 1)
    speed = np.minimum(norm / dt, 100)
    degree = np.bincount(trip.arcs.ravel(), minlength=len(trip.nodes))
    previous_distance, _ = project(trip.gps[prev, None, :], starts[ids], ends[ids])
    next_distance, _ = project(trip.gps[nxt, None, :], starts[ids], ends[ids])
    feature = np.stack([distances, np.broadcast_to(np.arange(k), ids.shape), cosine, sine,
                        trip.lengths[ids], fractions_out, np.broadcast_to(speed[:, None], ids.shape),
                        np.broadcast_to(dt[:, None], ids.shape), previous_distance, next_distance,
                        degree[trip.arcs[ids, 0]], degree[trip.arcs[ids, 1]]], axis=-1)
    return {'ids': ids, 'distances': distances, 'fractions': fractions_out,
            'features': feature.astype(np.float32)}


def load_reference(folder, trip):
    """Evaluation/training only: monotone geometric alignment of route to GPS.

    Route rows are traversed arcs, not per-observation labels. These aligned labels
    are proxy references, not independently verified pointwise ground truth.
    """
    path = Path(folder) / (trip.trip_id + '.route')
    route = np.loadtxt(path, dtype=int, ndmin=1)
    if not len(route) or route.min() < 0 or route.max() >= len(trip.arcs):
        raise ValueError('Invalid reference arc ID')
    starts = trip.xy[trip.arcs[route, 0]]
    ends = trip.xy[trip.arcs[route, 1]]
    back = np.zeros((len(trip.gps), len(route)), dtype=np.int32)
    cost = np.zeros(len(route))
    for i, point in enumerate(trip.gps):
        dist, _ = project(point, starts, ends)
        # O(T*R) prefix minimum recurrence allows arbitrary forward skips.
        best = np.minimum.accumulate(cost)
        change = np.r_[True, best[1:] < best[:-1]]
        predecessor = np.maximum.accumulate(np.where(change, np.arange(len(route)), 0))
        back[i] = predecessor
        cost = best + np.minimum(dist, 1000)**2
    cursor = int(np.argmin(cost)); indices = np.empty(len(trip.gps), dtype=int)
    for i in range(len(indices)-1, -1, -1):
        indices[i] = cursor; cursor = int(back[i, cursor])
    labels = route[indices]
    residual, _ = project(trip.gps, starts[indices], ends[indices])
    return labels, route, residual
