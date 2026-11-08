# ACPDAD

ACPDAD is a simulation-based research prototype for studying five modeled telemetry and network scenarios: **Benign, DoS, Replay, Evil_Twin, and FDI**. It replays simulated observations through a detector that combines a temporal classifier, a short-history fallback, physical-consistency warnings, and communication-health warnings.

All supplied data and reported results are simulated. This project does not demonstrate ArduPilot SITL, genuine Wi-Fi capture, a physical drone connection, or real-world attack detection.

## What is included

- A coupled simulated flight and network generator.
- A five-class detector with a 22-feature base input and 88 causal temporal features (window length 10).
- A three-observation fallback for startup and recovery after gaps.
- A selected checkpoint, its manifest, and compatible feature schema.
- Train/validation data and a frozen final evaluation set across five changed simulated conditions.
- Prediction ledgers, observation audits, per-mission metrics, dashboards, and historical comparison evidence.

Labels, attack schedules, run IDs, seeds, split assignments, and AP identities are excluded from model inputs. The runtime joins ground truth only after inference for scoring. Camera and MediaPipe inputs are not used.

## Requirements and setup

The project uses Python and the pinned packages in `requirements.txt`. The existing `.venv` is local to this workspace and is not part of the project ZIP. For a separate installation, create an environment and install the dependencies there:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.venv/matplotlib-cache"
```

The supplied checkpoint was verified with NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1, scikit-learn 1.9.1, and Matplotlib 3.11.2. The loader checks the checkpoint digest and feature contract before inference.

## Project layout

| Path | Contents |
| --- | --- |
| `run_project.py` | Single command-line entry point |
| `src/acpdad/` | Generator, runtime, inference, training, and evaluation protocol |
| `config/` | Simulation settings and frozen protocol specifications |
| `data/train_validation/` | Train and validation runs used to fit and select the detector |
| `data/validation/` | Five independent validation condition cohorts |
| `data/frozen_test/` | Five one-pass test condition cohorts; do not regenerate or rescore |
| `models/selected/` | Selected checkpoint and metadata |
| `results/selection/` | Training, threshold, fallback, and alert-policy selection records |
| `results/protocol_configs/` | Saved condition configurations for validation and final evaluation |
| `results/frozen_test/` | Frozen metrics, predictions, audits, report, and dashboard |
| `results/historical/` | Earlier evaluation evidence cited by the frozen report |
| `results/replays/` | New replay dashboards |
| `results/verification/` | Verification records and the cleanup report |
| `docs/relocation_manifest.json` | Path and hash record for the project reorganization |

Each dataset directory contains `run_manifest.csv`, `schema.json`, `checksums.json`, `experiment_config.json`, delivered and emitted packets, extracted features, ground truth, network flow, and true-state records. The five frozen test cohorts are mission initial-state, wind/background traffic, attack timing/duration/intensity, replay delay/source segment, and unseen AP/benign roaming.

## Verify the selected model

Run from the project root:

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib-cache"
.venv/bin/python run_project.py verify
```

Verification checks dataset checksums, checkpoint compatibility, feature reconstruction, causal window handling, fallback selection, gap reset, run isolation, and packet ordering.

## Replay a validation mission

Choose a run ID from the dataset manifest instead of relying on a hard-coded identifier:

```bash
RUN_ID=$(.venv/bin/python -c "import pandas as pd; m=pd.read_csv('data/train_validation/run_manifest.csv'); print(m.loc[(m.split=='val') & (m.scenario=='Evil_Twin'),'run_id'].iloc[0])")
OUT="results/replays/validation_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python run_project.py demo --data data/train_validation --model models/selected \
  --split val --run-id "$RUN_ID" --output "$OUT"
```

The command replays saved simulated observations. It does not generate a new mission or connect to a vehicle. Add `--realtime` to pace the replay. Open the printed `index.html` path for its dashboard.

## Predict one run or all runs

Predictions keep the selected run mapping and arrival order. Unknown run IDs are rejected with the dataset's available IDs. Prediction outputs contain no ground-truth labels or attack metadata.

```bash
RUN_ID=$(.venv/bin/python -c "import pandas as pd; m=pd.read_csv('data/train_validation/run_manifest.csv'); print(m.loc[(m.split=='val') & (m.scenario=='Evil_Twin'),'run_id'].iloc[0])")
.venv/bin/python run_project.py predict --data data/train_validation --model models/selected \
  --run-id "$RUN_ID" --output "results/predictions/selected_$(date -u +%Y%m%dT%H%M%SZ).csv"
.venv/bin/python run_project.py predict --data data/validation/mission_initial_state \
  --model models/selected --all-runs \
  --output "results/predictions/all_runs_$(date -u +%Y%m%dT%H%M%SZ).csv"
```

## Generate a new simulated dataset

Generation writes to a new directory and leaves the supplied datasets unchanged:

```bash
.venv/bin/python run_project.py generate --config config/simulation.json --seed 9654321 \
  --runs-per-class 1 --duration 30 --profile patrol --output data/generated/seed_9654321
RUN_ID=$(.venv/bin/python -c "import pandas as pd; m=pd.read_csv('data/generated/seed_9654321/run_manifest.csv'); print(m.loc[m.scenario=='Evil_Twin','run_id'].iloc[0])")
.venv/bin/python run_project.py predict --data data/generated/seed_9654321 --model models/selected \
  --run-id "$RUN_ID" --output results/predictions/generated_$(date -u +%Y%m%dT%H%M%SZ).csv
```

Options include `--seed`, `--runs-per-class`, `--duration`, `--hz`, `--profile`, and repeatable `--set KEY=JSON`. The generator creates synthetic missions, not measured flights.

## Reproduce model selection

Training fits on training missions and selects settings using validation data. Use fresh output directories; training refuses to overwrite them.

```bash
.venv/bin/python run_project.py train --data data/train_validation \
  --model models/reproductions/run_001 --results results/reproductions/run_001 \
  --validation-data data/validation/mission_initial_state \
  --validation-data data/validation/wind_background_traffic \
  --validation-data data/validation/attack_timing_duration_intensity \
  --validation-data data/validation/replay_delay_source_segment \
  --validation-data data/validation/unseen_ap_benign_roaming
```

Alert calibration may be run against a reproduction model with a separate output directory. Do not point training or calibration at `models/selected/`; it contains the frozen selected checkpoint. Use `run_project.py evaluate --split val` with a fresh output directory for validation evaluation.

```bash
.venv/bin/python run_project.py calibrate-alerts --data data/train_validation \
  --model models/reproductions/run_001 \
  --results results/reproductions/run_001/alert_calibration \
  --validation-data data/validation/mission_initial_state \
  --validation-data data/validation/wind_background_traffic \
  --validation-data data/validation/attack_timing_duration_intensity \
  --validation-data data/validation/replay_delay_source_segment \
  --validation-data data/validation/unseen_ap_benign_roaming
.venv/bin/python run_project.py evaluate --data data/validation/wind_background_traffic \
  --model models/reproductions/run_001 --split val \
  --output results/reproductions/run_001/evaluation
```

The supplied final test has already been scored once. Its lock and predictions are preserved; `evaluate-final` refuses a second scoring pass. The saved final condition configurations are under `results/protocol_configs/final/`.

## Results and limitations

The frozen simulated evaluation covers 150,000 scheduled observations. Timely prediction coverage was **92.54%**. Evil_Twin recall was **98.75%** on timely scored rows and **84.76%** across all scheduled attack rows; pooled precision was **98.09%**. Benign scored-row false-positive rate was **3.38%**. Five-class scored macro-F1 was **0.866**, with DoS recall at **40.65%**. The unseen-AP/benign-roaming Evil_Twin precision was **93.75%**. Benign alert missions were **36%**, and false alert episodes were **0.79 per minute**. Several prespecified targets were missed.

Open the [frozen report](results/frozen_test/IMPLEMENTATION_REPORT.md), [metrics](results/frozen_test/analysis/metrics.json), [per-mission results](results/frozen_test/analysis/per_mission_metrics.csv), [complete observation audit](results/frozen_test/analysis/all_observation_audit.csv), or [dashboard](results/frozen_test/index.html). The [results index](results/README.md) explains the evidence folders and the archived pre-correction report copy.

The flight and network simulator uses simplified dynamics. Its queue-priority behavior is a modeling assumption rather than measured radio scheduling. Results support claims only for the simulated conditions tested; they do not establish field, physical-drone, or real-world performance.
