import numpy as np
from scipy.sparse.csgraph import dijkstra


def transition_distances(trip, cand, t):
    """Directed shortest network travel between projected candidate positions."""
    cache = cand.setdefault('_transitions', {})
    if t in cache: return cache[t]
    previous, current = cand['ids'][t-1], cand['ids'][t]
    sources = trip.arcs[previous, 1]
    targets = trip.arcs[current, 0]
    unique, inverse = np.unique(sources, return_inverse=True)
    gps_step = float(np.linalg.norm(trip.gps[t]-trip.gps[t-1]))
    dt = max(float(trip.track[t, 2]-trip.track[t-1, 2]), 1)
    limit = max(500.0, gps_step*4 + 200, min(dt*70, 20000))
    road = dijkstra(trip.graph, directed=True, indices=unique, limit=limit)
    remaining = (1-cand['fractions'][t-1])*trip.lengths[previous]
    progress = cand['fractions'][t]*trip.lengths[current]
    distance = remaining[:, None] + road[inverse][:, targets] + progress[None, :]
    same = previous[:, None] == current[None, :]
    forward = cand['fractions'][t][None, :] >= cand['fractions'][t-1][:, None] - 0.02
    direct = np.maximum(0, progress[None, :] - (cand['fractions'][t-1]*trip.lengths[previous])[:, None])
    distance = np.where(same & forward, np.minimum(distance, direct), distance)
    cache[t] = (distance, gps_step)
    return cache[t]


def decode(trip, cand, probabilities, beta=35.0, gap_seconds=30,
           transition_weight=1.0, normalize_transitions=True):
    """Log-space Viterbi with normalized directed transition probabilities.

    A gap or a step with no reachable candidate pair starts a new route component.
    No impossible connector is fabricated. Reset indices are returned for the UI.
    """
    if beta <= 0 or transition_weight <= 0:
        raise ValueError('Decoder scales must be positive')
    n, k = probabilities.shape
    emission = np.log(np.clip(probabilities, 1e-12, 1))
    score = emission[0].copy()
    back = np.zeros((n, k), dtype=np.int32)
    resets = [0]
    for t in range(1, n):
        distance, gps_step = transition_distances(trip, cand, t)
        transition = -np.abs(distance-gps_step) / beta
        if normalize_transitions:
            finite = np.isfinite(transition)
            row_max = np.max(np.where(finite, transition, -1e30), axis=1)
            denominator = np.sum(np.where(finite, np.exp(np.clip(transition-row_max[:, None], -700, 0)), 0), axis=1)
            transition = np.where(finite, transition-row_max[:, None]-np.log(np.maximum(denominator[:, None], 1e-300)), -np.inf)
        transition *= transition_weight
        combined = score[:, None] + transition
        if trip.track[t, 2]-trip.track[t-1, 2] > gap_seconds or not np.isfinite(combined).any():
            # A reset permits independent decoding, and is visibly marked as disconnected.
            back[t] = int(np.argmax(score)); score = emission[t].copy(); resets.append(t)
        else:
            back[t] = np.argmax(combined, axis=0)
            score = combined[back[t], np.arange(k)] + emission[t]
        score -= np.max(score)
    selected = np.empty(n, dtype=int); selected[-1] = int(np.argmax(score))
    for t in range(n-1, 0, -1): selected[t-1] = back[t, selected[t]]
    return cand['ids'][np.arange(n), selected], selected, resets


def continuity(trip, cand, chosen):
    if len(chosen) < 2: return 1.0
    good = 0
    for t in range(1, len(chosen)):
        distances, _ = transition_distances(trip, cand, t)
        good += int(np.isfinite(distances[chosen[t-1], chosen[t]]))
    return good/(len(chosen)-1)


def route_geometry(trip, cand, chosen, resets=()):
    """Expand directed shortest connectors and preserve separate route components."""
    components, component, route_arcs = [], [], []
    reset_set = set(resets)
    previous_arc = None; previous_frac = 0.0
    for t, col in enumerate(chosen):
        arc = int(cand['ids'][t, col]); frac = float(cand['fractions'][t, col])
        u, v = map(int, trip.arcs[arc])
        point = trip.nodes[u] + frac*(trip.nodes[v]-trip.nodes[u])
        if previous_arc is not None and t not in reset_set:
            pu, pv = map(int, trip.arcs[previous_arc])
            if arc == previous_arc and frac >= previous_frac-0.02:
                component.append(point[::-1].tolist())
            else:
                dist, predecessor = dijkstra(trip.graph, directed=True, indices=pv, return_predecessors=True,
                                             limit=20000)
                if np.isfinite(dist[u]):
                    chain=[u]; cur=u
                    while cur != pv:
                        cur=int(predecessor[cur]); chain.append(cur)
                    component.extend(trip.nodes[chain[::-1], ::-1].tolist())
                    component.append(point[::-1].tolist())
                else:
                    if component: components.append(component)
                    component=[point[::-1].tolist()]
        else:
            if component: components.append(component)
            component=[point[::-1].tolist()]
        route_arcs.append(arc); previous_arc=arc; previous_frac=frac
    if component: components.append(component)
    return components, route_arcs
