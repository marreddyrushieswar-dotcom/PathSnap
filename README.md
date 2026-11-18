# PathSnap

GPS map matching using candidate classification and directed HMM/Viterbi route reconstruction.

## Deliverable 1

| Folder | Contents |
|---|---|
| pathsnap/ | Input processing, candidate features, five-model experiments, Bayesian search, decoder and API |
| ui/ | Browser interface for the receiving laptop |
| artifacts/ | Five trained classifiers, selected model and inference configuration |
| results/ | Measured comparisons, validation folds, search trials, predictions and API checks |
| reports/ | EDA figures, measured comparisons and geographical visualisations |
| scripts/ | Decoder tuning, EDA/comparison plotting, map generation and validation |
| tests/ | Direction, GPS gap, reference alignment and decoder correctness checks |
| runs/baseline_v1/ | Recorded configuration and results from the original baseline |
| vendor/ | Third-party visualisation dependencies, source provenance and licenses |

Deliverable 2 is submitted separately. The legacy report template describes the original baseline and is guarded against generating an inaccurate report for the updated pipeline.

## Install and run

Python 3.11 is recommended. Tested dependencies are recorded in `results/environment.txt`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start_api.ps1 -BindAddress 0.0.0.0
```

Place the official dataset at `../Student folder`, containing `train/` and `student_test/`, or set `PATHSNAP_DATA_ROOT` to its location. Open `http://127.0.0.1:8000` on the model laptop. On the second laptop on the same trusted network, open `http://MODEL_LAPTOP_IP:8000`, enter the same server URL, connect and select a trip. Port 8000 must be accessible.

Endpoints: `GET /health`, `GET /api/trips`, `POST /api/predict`. Interactive API documentation is at `/docs`. Set `PATHSNAP_EVALUATOR_ROOT` only when organisers supply reserved trips. Each trip requires matching `.track`, `.nodes`, `.arcs` and optional server-only `.route` files. Model files must be loaded with compatible scikit-learn versions or retrained.

## Updated feature pipeline

The candidate set remains up to 12 directed road segments per observation. Feature schema 2 contains 40 features:

- The original 12 distance, rank, heading, segment length, projection, speed, timing and endpoint-connectivity features.
- Four relative-distance and segment-endpoint-distance features.
- Eight context features at each of three time scales (3, 8 and 20 seconds): direction alignment, displacement, speed, past/future distance to the road, mean GPS position distance, projected progress and elapsed time.

Context does not cross GPS gaps longer than 30 seconds. This is offline route reconstruction: future GPS observations are available. References and arc IDs are not predictive features. The original 12-feature schema remains supported for old model artifacts. Loading a mismatched model/configuration fails explicitly.

## Training and model selection

The five classifiers are Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest and Histogram Gradient Boosting. Updated experiments sample at most 900 observations per training trip, with seed 181126. GroupKFold keeps complete trips apart in three folds; learned scaling is fitted within each fold. Classifier selection and Bayesian tuning use grouped-validation macro F1. The chosen model is refitted on the 80 training trips.

The decoder search fits a separate classifier on 64 training trips and selects sequence parameters on the remaining 16 complete training trips. It compares transition scales and relative weights using normalised directed transitions, selecting observation-weighted accuracy with macro F1 as a tie-breaker. These are tuning results, not an independent estimate of generalisation. Student-test labels are not used for this search. The final classifier and decoder are evaluated on all observations in all ten student-test trips.

See `results/improvement_comparison.json`, `results/overall_metrics.json`, `results/model_comparison.csv`, `results/decoder_comparison.csv`, `results/bayesian_trials.json` and `results/decoder_search.json` for actual measurements. The comparison changes features, sampling and model settings together; it is not a feature-only ablation. Reserved evaluator data was not supplied or used. Results above 90% on student development data do not guarantee the same result on unseen reserved trips or on every individual trip.

## Reproduce

From an activated environment:

```powershell
$env:OMP_NUM_THREADS='2'
python -m pathsnap.experiment --data '..\Student folder' --feature-version 2 --max-observations 900 --trials 12 --output runs\context_v2
python scripts\tune_decoder.py --run runs\context_v2
python scripts\plot_experiments.py --run runs\context_v2
python scripts\compare_runs.py --baseline runs\baseline_v1 --candidate runs\context_v2
python -m unittest discover -s tests -v
```

To test a completed experiment before deployment, set `PATHSNAP_ARTIFACT_ROOT` to its `artifacts/` directory before starting the API. Restart the API after changing artifacts: the loaded model is cached. Regenerate geographical maps using `python scripts/make_maps.py` after deploying the matching predictions. Temporary prepared-data caches and duplicate experiment artifacts are excluded from Git.

## Metrics and provenance

The `.route` files describe traversed arc sequences rather than native GPS-point annotations. A monotone geometric alignment derives labels for training and scoring. This definition is unchanged from the baseline. Pointwise accuracy means agreement with these derived reference labels and can be optimistic because geometry also appears in features.

Accuracy and log loss are observation weighted; macro F1, precision and recall are calculated per trip and averaged. Missing reference candidates count as errors. Route continuity measures directed reachability within the documented search limit; arc-set F1 measures coverage rather than traversal order. Candidate scores are normalised heuristic HMM emissions without guaranteed calibration. Local distances use an equirectangular approximation. The decoder starts separate components for large GPS gaps or unreachable transitions.

Prediction reads only GPS and network inputs. Reference routes are opened afterwards for server-side scoring and are not returned to the UI. Handler checks, live HTTP evidence and two-laptop checks have separate verification scopes; old screenshots are evidence of the baseline until the updated server is demonstrated.

AI assistance was used for implementation and drafting. Measurements come from executed code. Third-party sources retain their licenses and attribution.

`results/run_manifest.json` records input-integrity and prediction checks. `results/submission_sha256.json` hashes Git-normalised file contents (excluding that checksum file itself).
