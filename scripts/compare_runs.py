"""Compare a completed candidate run with the currently deployed baseline."""
import argparse
import csv
import json
from pathlib import Path
from plot_experiments import bars


def summarize(root):
    config = json.loads((root/'artifacts'/'config.json').read_text())
    with (root/'results'/'dataset_inventory.csv').open() as source:
        sizes = {row['trip_id']: int(row['observations']) for row in csv.DictReader(source) if row['split']=='student_test'}
    with (root/'results'/'decoder_comparison.csv').open() as source:
        rows = list(csv.DictReader(source))
    result = dict(feature_count=len(config['features']), model=config['selected_model'],
                  training_observations=config['training_observations'],
                  selected_cv_macro_f1=max(config['untuned_cv_macro_f1'],config['bayesian_cv_macro_f1']),
                  observations=sum(sizes.values()), label_method=config['label_method'])
    for variant in ('classifier','hmm'):
        selected = [row for row in rows if row['variant']==variant]
        if len(selected)!=len(sizes) or {row['trip_id'] for row in selected}!=set(sizes):
            raise ValueError('Comparison requires every student trip exactly once per variant')
        result[variant] = dict(accuracy=sum(float(row['accuracy'])*sizes[row['trip_id']] for row in selected)/sum(sizes.values()),
                               macro_f1=sum(float(row['macro_f1']) for row in selected)/len(selected),
                               mean_route_continuity=sum(float(row['route_continuity']) for row in selected)/len(selected))
    return result


def main(baseline, candidate):
    before, after = summarize(baseline), summarize(candidate)
    if before['observations']!=after['observations'] or before['label_method']!=after['label_method']:
        raise ValueError('Baseline and candidate must use identical observations and label definitions')
    result = dict(baseline=before, candidate=after,
                  accuracy_gain_percentage_points=100*(after['hmm']['accuracy']-before['hmm']['accuracy']),
                  target_above_90_percent=after['hmm']['accuracy']>.9,
                  notes=['All ten student-test trips; observation-weighted accuracy.',
                         'Labels and scoring definition unchanged.',
                         'Feature set, sample count and model configuration changed together; this is not a feature-only ablation.',
                         'Not a result on reserved evaluator trips.'])
    (candidate/'results'/'improvement_comparison.json').write_text(json.dumps(result,indent=2))
    bars(candidate/'reports'/'before_after_accuracy.png', 'All student-test observations: agreement with derived reference labels',
         ['Baseline (12 features)',f'Updated ({after["feature_count"]} features)'],
         [[entry['classifier']['accuracy'] for entry in (before,after)],
          [entry['hmm']['accuracy'] for entry in (before,after)]],
         ['Classifier only','Classifier + HMM'],1)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args()
    main(args.baseline.resolve(),args.candidate.resolve())
