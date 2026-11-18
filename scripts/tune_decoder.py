"""Tune sequence decoding on held-out TRAINING trips, then evaluate student trips.

The classifier used for decoder selection never fits the held-out trips. The
deployed classifier is refitted on all training trips by pathsnap.experiment.
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pathsnap import experiment as exp
from pathsnap.data import load_trip
from pathsnap.decoder import decode, continuity, route_geometry


def cached(run, split, trip_id):
    matches = list((run/'artifacts'/'prepared').glob(f'{split}_{trip_id}_k12_v2_*.npz'))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one matching cache for {split}/{trip_id}')
    with np.load(matches[0]) as data:
        return {key: data[key] for key in data.files}


def main(run):
    config = json.loads((run/'artifacts'/'config.json').read_text())
    data_root = Path(config['data_root'])
    rng = np.random.default_rng(exp.SEED)
    parts = []
    for folder in sorted((data_root/'train').iterdir()):
        if not folder.is_dir():
            continue
        c = cached(run, 'train', folder.name)
        take = np.sort(rng.choice(len(c['labels']), min(config['maximum_training_observations_per_trip'], len(c['labels'])), replace=False))
        parts.append((c['features'][take], c['ids'][take], c['labels'][take], np.full(len(take), folder.name)))
    x, ids, labels, groups = [np.concatenate([part[i] for part in parts]) for i in range(4)]
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=exp.SEED).split(x, labels, groups))
    validation_trips = sorted(np.unique(groups[val]).tolist())
    model = clone(joblib.load(run/'artifacts'/'final_model.joblib'))
    exp.fit(model, x[train], ids[train], labels[train])
    grid = [dict(beta=beta, gap_seconds=30, transition_weight=weight, normalize_transitions=normalise)
            for beta, weight, normalise in itertools.product((15., 50., 150.), (.15, .5, 1.), (True,))]
    records = [dict(parameters=parameters, correct=0, observations=0, trip_f1=[]) for parameters in grid]
    baseline = dict(correct=0, observations=0)
    for trip_id in validation_trips:
        trip = load_trip(data_root/'train'/trip_id)
        c = cached(run, 'train', trip_id)
        probability = exp.scores(model, c['features'])
        raw = c['ids'][np.arange(len(c['labels'])), probability.argmax(axis=1)]
        baseline['correct'] += int(np.sum(raw == c['labels']))
        baseline['observations'] += len(raw)
        for record in records:
            pred, _, _ = decode(trip, c, probability, **record['parameters'])
            record['correct'] += int(np.sum(pred == c['labels']))
            record['observations'] += len(pred)
            record['trip_f1'].append(float(f1_score(c['labels'], pred, average='macro', zero_division=0)))
        print(f'Decoder training holdout: {trip_id}', flush=True)
    for record in records:
        record['accuracy'] = record['correct']/record['observations']
        record['macro_f1'] = float(np.mean(record.pop('trip_f1')))
    best = max(records, key=lambda record: (record['accuracy'], record['macro_f1']))
    search = dict(selection_partition='16 held-out training trips; not student_test',
                  primary_metric='observation_weighted_accuracy', seed=exp.SEED,
                  fit_trips=sorted(np.unique(groups[train]).tolist()), validation_trips=validation_trips,
                  classifier_accuracy=baseline['correct']/baseline['observations'], trials=records,
                  chosen=best)
    (run/'results'/'decoder_search.json').write_text(json.dumps(search, indent=2))
    config['decoder'] = best['parameters']
    config['decoder_validation_accuracy'] = best['accuracy']
    (run/'artifacts'/'config.json').write_text(json.dumps(config, indent=2))
    final = joblib.load(run/'artifacts'/'final_model.joblib')
    rows = []
    totals = {key: dict(correct=0, observations=0, f1=[]) for key in ('classifier', 'hmm')}
    for folder in sorted((data_root/'student_test').iterdir()):
        if not folder.is_dir():
            continue
        trip = load_trip(folder)
        c = cached(run, 'student_test', folder.name)
        probability = exp.scores(final, c['features'])
        start = time.perf_counter()
        pred, chosen, resets = decode(trip, c, probability, **config['decoder'])
        seconds = time.perf_counter()-start
        for variant, columns in [('classifier', probability.argmax(axis=1)), ('hmm', chosen)]:
            prediction = c['ids'][np.arange(len(pred)), columns]
            correct = int(np.sum(prediction == c['labels']))
            f1 = float(f1_score(c['labels'], prediction, average='macro', zero_division=0))
            predicted_set, reference_set = set(prediction.tolist()), set(c['route'].tolist())
            rows.append(dict(trip_id=folder.name, variant=variant, observations=len(pred),
                             accuracy=correct/len(pred), macro_f1=f1,
                             route_continuity=continuity(trip, c, columns),
                             route_arc_set_f1=2*len(predicted_set & reference_set)/max(len(predicted_set)+len(reference_set), 1),
                             resets=len(resets)-1 if variant=='hmm' else 0,
                             decode_seconds=seconds if variant=='hmm' else 0))
            totals[variant]['correct'] += correct
            totals[variant]['observations'] += len(pred)
            totals[variant]['f1'].append(f1)
        geometry, arcs = route_geometry(trip, c, chosen, resets)
        output = dict(trip_id=folder.name, gps=trip.track[:, :2][:, ::-1].tolist(),
                      predicted_components=geometry, predicted_arc_ids=arcs,
                      original_rows=trip.original_rows.tolist(), resets=resets,
                      accuracy=float(np.mean(pred == c['labels'])),
                      reference_kind='monotone-geometric proxy alignment')
        (run/'results'/'predictions'/(folder.name+'.json')).write_text(json.dumps(output))
        print(f'Student trip {folder.name}: {output["accuracy"]:.4%}', flush=True)
    exp.write_csv(run/'results'/'decoder_comparison.csv', rows)
    for total in totals.values():
        total['accuracy'] = total['correct']/total['observations']
        total['macro_f1'] = float(np.mean(total.pop('f1')))
        total['target_above_90_percent'] = total['accuracy'] > .9
    totals['accuracy_basis'] = config['label_method']
    totals['decoder_selection'] = 'training holdout only'
    (run/'results'/'overall_metrics.json').write_text(json.dumps(totals, indent=2))
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
