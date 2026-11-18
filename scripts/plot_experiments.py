"""Generate EDA and comparison figures from recorded data, without creating a report."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def font(size):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    except OSError:
        return ImageFont.load_default()


def bars(path, title, labels, series, names, maximum=None):
    canvas = Image.new('RGB', (1400, 780), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 24), title, fill='#17354a', font=font(28))
    left, top, width, height = 105, 110, 1240, 460
    maximum = maximum or max(max(values) for values in series)*1.1 or 1
    palette = ['#167d8d', '#da8737', '#6379b3']
    for tick in range(6):
        y = top+height-tick*height/5
        draw.line((left, y, left+width, y), fill='#d6dfe5')
        draw.text((12, y-10), f'{maximum*tick/5:.2f}', fill='#30485a', font=font(18))
    group = width/len(labels)
    bar = .75*group/len(series)
    for index, label in enumerate(labels):
        for j, values in enumerate(series):
            value = values[index]
            x = left+group*index+.1*group+j*bar
            y = top+height-value/maximum*height
            draw.rectangle((x, y, x+.9*bar, top+height), fill=palette[j % 3])
            draw.text((x, max(y-24, 70)), f'{value:.3f}', fill='#17354a', font=font(17))
        draw.multiline_text((left+group*index+6, top+height+15), label, fill='#17354a', font=font(18))
    for j, name in enumerate(names):
        draw.rectangle((left+j*390, 690, left+j*390+20, 710), fill=palette[j % 3])
        draw.text((left+j*390+30, 684), name, fill='#17354a', font=font(21))
    canvas.save(path)


def histogram(path, title, values, bins, unit):
    counts, edges = np.histogram(values, bins=bins)
    labels = [f'{edges[i]:g}\nto\n{edges[i+1]:g}' for i in range(len(counts))]
    bars(path, title, labels, [counts.tolist()], [f'Count ({unit})'])


def main(run):
    out = run/'reports'
    out.mkdir(exist_ok=True, parents=True)
    with (run/'results'/'model_comparison.csv').open() as source:
        models = list(csv.DictReader(source))
    labels = [row['model'].replace('Regression', '\nRegression').replace('Forest', '\nForest').replace('Gradient', '\nGradient').replace('Boosting', '\nBoosting') for row in models]
    bars(out/'cv_comparison.png', 'Trip-grouped validation macro F1', labels,
         [[float(row['cv_macro_f1']) for row in models]], ['Macro F1'], 1)
    bars(out/'student_comparison.png', 'Student-test classifier metrics (derived reference labels)', labels,
         [[float(row[key]) for row in models] for key in ('top1_accuracy', 'macro_f1')], ['Top-1 accuracy', 'Mean per-trip macro F1'], 1)
    bars(out/'logloss_comparison.png', 'Student-test candidate log loss', labels,
         [[float(row['log_loss']) for row in models]], ['Log loss (lower is better)'])
    trials = json.loads((run/'results'/'bayesian_trials.json').read_text())
    bars(out/'bayesian_search.png', 'Measured Bayesian optimisation trials', [str(row['trial']+1) for row in trials],
         [[row['cv_macro_f1'] for row in trials]], ['Trip-grouped macro F1'], 1)
    config = json.loads((run/'artifacts'/'config.json').read_text())
    observations, intervals, speeds, segment_lengths, issues = [], [], [], [], []
    for folder in sorted((Path(config['data_root'])/'train').iterdir()):
        if not folder.is_dir():
            continue
        # EDA needs NumPy only; it does not load a model or alter training data.
        raw=np.loadtxt(folder/(folder.name+'.track'),ndmin=2)
        nodes=np.loadtxt(folder/(folder.name+'.nodes'),ndmin=2)
        arcs=np.loadtxt(folder/(folder.name+'.arcs'),dtype=int,ndmin=2)
        valid=np.isfinite(raw).all(axis=1)&(np.abs(raw[:,:2])<=[180,90]).all(axis=1)
        track=raw[valid]
        out_of_order=int((np.diff(track[:,2])<0).sum())
        track=track[np.argsort(track[:,2],kind='stable')]
        origin=nodes.mean(axis=0)
        scale=np.array([np.cos(np.deg2rad(origin[1])),1.])*np.pi/180*6371008.8
        gps=(track[:,:2]-origin)*scale
        xy=(nodes-origin)*scale
        lengths=np.linalg.norm(xy[arcs[:,1]]-xy[arcs[:,0]],axis=1)
        dt = np.diff(track[:, 2])
        distance = np.linalg.norm(np.diff(gps, axis=0), axis=1)
        observations.append(len(track))
        intervals.extend(dt.tolist())
        speeds.extend((distance[dt > 0]/dt[dt > 0]).tolist())
        segment_lengths.extend(lengths.tolist())
        issues.append(dict(invalid_observations=int((~valid).sum()),out_of_order_steps=out_of_order,
                           duplicate_timestamps=int((dt==0).sum()),gaps_over_10s=int((dt>10).sum()),
                           zero_length_arcs=int((lengths<1e-6).sum())))
    histogram(out/'eda_sampling_gaps.png', 'TRAIN only: sampling intervals in seconds', intervals, [0, 1, 2, 5, 10, 30, 60, max(61, max(intervals)+1)], 'sampling intervals')
    histogram(out/'eda_speeds.png', 'TRAIN only: raw GPS step speed (metres per second)', speeds, [0, 1, 5, 10, 20, 40, 70, 100, max(101, max(speeds)+1)], 'steps with positive time')
    histogram(out/'eda_trip_sizes.png', 'TRAIN only: observations per trip', observations, 10, 'trips')
    fields = ['invalid_observations', 'out_of_order_steps', 'duplicate_timestamps', 'gaps_over_10s', 'zero_length_arcs']
    bars(out/'eda_data_quality.png', 'TRAIN only: recorded input-quality issues', [field.replace('_', '\n') for field in fields],
         [[sum(row[field] for row in issues) for field in fields]], ['Count'])
    summary = dict(partition='train', trips=len(observations), observations=sum(observations),
                   sampling_interval_median_s=float(np.median(intervals)),
                   raw_speed_median_mps=float(np.median(speeds)),
                   raw_speed_p99_mps=float(np.percentile(speeds, 99)),
                   segment_length_median_m=float(np.median(segment_lengths)),
                   issues={field: sum(row[field] for row in issues) for field in fields})
    (run/'results'/'eda_summary.json').write_text(json.dumps(summary, indent=2))
    # Reuse the baseline's geographical drawing helper, not its report text.
    # Keeping the same representative trip makes the before/after view comparable.
    import build_report as geographical_drawing
    geographical_drawing.OUT=out
    folder=Path(config['data_root'])/'student_test'/'00000052'
    nodes=np.loadtxt(folder/'00000052.nodes',ndmin=2)
    arcs=np.loadtxt(folder/'00000052.arcs',dtype=int,ndmin=2)
    route=np.loadtxt(folder/'00000052.route',dtype=int,ndmin=1)
    prediction=json.loads((run/'results'/'predictions'/'00000052.json').read_text())
    geographical_drawing.trajectory_image(prediction,nodes,arcs,route)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=PROJECT)
    main(parser.parse_args().run.resolve())
