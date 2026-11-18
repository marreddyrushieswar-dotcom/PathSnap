# Development status

Completed and executed:

- Isolated Python 3.11 environment and required ML/API dependencies.
- Inventory and quality checks across all 80 training and 10 student-test trips.
- Label-free feature extraction and directed road candidate generation.
- Five fitted classifiers with trip-grouped cross-validation and full student-test evaluation.
- Ten Gaussian-process Bayesian tuning trials and final fitted model.
- HMM/Viterbi decoding on every student-test trip, with measured comparisons.
- FastAPI backend, evaluator-facing UI, and two-laptop startup instructions.
- Unit tests and direct API-handler validation on all ten student-test trips.
- Folium development map, measured plots and a consolidated technical report.

Not verified or submitted:

- Live HTTP service, event-loop startup, and browser-to-API connection: this
  execution environment restricts sockets, including asyncio socketpair.
- Rendered UI/Folium screenshots: browser policy blocks local file navigation.
- Actual two-laptop network deployment and receiving-laptop log screenshots.
- Final evaluation on the ten reserved trips, which were not supplied.
- Publication to the team's public GitHub repository and final report submission.
- Team identity and repository URL in the final submission.

Pointwise metrics use derived geometric route alignment, not supplied native
GPS-point labels. All saved model and decoder measurements are from actual
executed experiments. AI assistance is disclosed in the README and report.
