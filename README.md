# PathSnap

GPS map matching using candidate classification and directed HMM/Viterbi decoding.
The working data directory is `../Student folder`, containing `train` (80 trips)
and `student_test` (10 trips). Reserved evaluator data was not supplied or used.

## Quick start on this computer

From this folder in PowerShell:

```powershell
..\.venv311\Scripts\python.exe -m unittest discover -s tests -v
.\start_api.ps1
```

Open http://127.0.0.1:8000 for the UI, or http://127.0.0.1:8000/docs for API documentation.
`GET /health`, `GET /api/trips`, and `POST /api/predict` are the service endpoints.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/predict -Method Post -ContentType application/json -Body '{"trip_id":"00000000"}'
```

## Clean installation

Use a supported Python 3.11 interpreter. The default Python launcher on the
development computer was not configured; a local Python 3.11 environment was
created separately without changing system PATH or global packages.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pathsnap.experiment --data '..\Student folder' --trials 10
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000
```

Model pickles require compatible Python/scikit-learn versions. Prefer retraining
when moving to a different environment. Load only the trusted model files produced
by this project. `results/environment.txt` records the executed environment.

## Two-laptop demonstration

1. Connect both laptops to the same trusted network.
2. On Laptop 1, start `./start_api.ps1 -BindAddress 0.0.0.0`.
3. On Laptop 2, open `ui/index.html` through a local HTTP server, for example
   `python -m http.server 8080 --directory ui`, and visit http://localhost:8080.
4. Enter `http://LAPTOP_1_IP:8000` in the UI and press Connect.
5. Choose a trip and reconstruct it. API logs on Laptop 1 show the request and result.

Alternatively Laptop 2 can open `http://LAPTOP_1_IP:8000`, which serves the same UI.
Network access depends on the operating system firewall; no firewall rules are
changed by this project. The UI contains no model or reference data.

## Reserved evaluator trips

Set the evaluator root only when organisers provide the reserved partition:

```powershell
$env:PATHSNAP_EVALUATOR_ROOT='C:\path\to\evaluator_test'
.\start_api.ps1 -BindAddress 0.0.0.0
```

Each trip directory must have matching `.track`, `.nodes`, `.arcs`, and optional
server-only `.route` files. The evaluator UI lists the actual available trip IDs.
Until then, the UI explicitly identifies the supplied student-development trips
as **Student demo**. It does not pretend they are reserved evaluator trips.

Prediction loads only `.track`, `.nodes`, and `.arcs`. Scoring opens `.route`
after prediction has finished and returns aggregate accuracy only. Neither
reference geometry nor reference arc labels are returned by the API.

## Method and metric definitions

- Local equirectangular coordinates in metres at each network's mean latitude.
  This approximates local distances; it is not a globally accurate projection.
- Segment samples at <=30 m feed a KD tree; the shortlist is reranked using
  exact point-to-segment distances. Twelve directed arc candidates per observation.
- Twelve features describe distance, heading, segment length, projection fraction,
  local speed/time, nearby observations, and endpoint degree. No trip-local arc ID
  is used as a learned feature, so the classifier can handle unseen networks.
- The `.route` file is a chronological sequence of traversed arcs, not GPS-point
  labels. A monotone geometric alignment derives training and scoring labels.
  Reported pointwise scores are against these **proxy reference labels** and can
  be optimistic because geometric information also appears in the model features.
- At most 350 uniformly random observations per training trip are fitted, with
  fixed seed 181126. Every one of the 80 training trips participates. Student
  metrics use all valid observations in all ten development trips.
- Three-fold GroupKFold keeps complete trips apart. Scaling is fitted inside
  each training fold. Primary selection metric: mean per-trip observation-level
  macro F1. Student-test results are reported but not used to select the model.
- Arc IDs are local to a network. Macro F1, precision and recall are calculated
  separately per trip and then averaged. Accuracy and log loss are observation
  weighted. Candidate misses are included as errors; log loss uses an epsilon
  penalty when the reference arc is missing.
- Candidate classification is binary: candidate matches aligned arc or not.
  Positive scores are normalized across candidates. These scores are heuristic
  HMM emissions; calibration is not guaranteed.
- Bayesian search uses a Gaussian-process surrogate with Matern kernel and
  expected improvement, after three random initialization points. The full
  search history and trip-fold assignments are saved.
- Viterbi uses directed shortest-path distances between projected candidates.
  Log transition scores penalize the discrepancy between network and GPS travel.
  Gaps over 30 seconds and completely unreachable steps start new components.
  No straight-line connector is fabricated between disconnected components.
- Route continuity is the fraction of adjacent selected states reachable on the
  directed network within the documented search limit. Arc-set F1 measures
  reference-route arc coverage; it does not measure traversal order.

## Project layout

| Folder | Contents |
|---|---|
| pathsnap/ | Loading, features, experiments, decoder, backend |
| ui/ | Offline geographical plot and evaluator interface |
| tests/ | Directed transitions, gaps, alignment and input handling |
| artifacts/ | Five fitted models, final model and inference configuration |
| results/ | Inventory, metrics, CV splits, Bayesian trials, predictions and evidence |
| reports/ | Generated technical report and figures |
| scripts/ | Report and geographical visualisation generation |
| vendor/ | Upstream visualization sources, licenses and source provenance |

`artifacts/prepared` is a reproducible local cache, excluded from the submission
archive. No evaluator data is needed to reproduce development experiments.

## Reproducibility and submission

Run `python -m pathsnap.experiment` to regenerate measured experiments. Training
and validation data, random seed, model configuration and trial records are
retained. Runtime figures vary by machine.

The required public GitHub repository, receiving-laptop demonstration, team
identity and access to organiser-held evaluator trips remain external submission
steps. Do not claim those were completed from a single-computer development run.
The execution session restricts sockets, including asyncio's internal socketpair.
HTTP transport and event-loop startup could not be verified. `scripts/validate_api.py`
executes synchronous API handlers on all ten student trips and records the exact
verification scope; it is not a substitute for a live two-laptop check.
Review and understand the source, findings and report before submission. AI
assistance was used for implementation and drafting; experimental measurements
are generated by executed code, not invented values.
