# ACPDAD V5.4R1 Evil_Twin detection implementation and evaluation

All data and outputs in this report are **simulated**. This is not ArduPilot SITL, a physical drone connection, captured Wi-Fi, or evidence about real-world attack performance.

The selected HistGradientBoosting model uses `max_iter=160`, `max_leaf_nodes=19`, `min_samples_leaf=25`, and `l2_regularization=1.0`; its Evil_Twin threshold was chosen on validation data. The `.venv` training environment used Python 3.14.9, NumPy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, and Matplotlib 3.11.2. Installed and training scikit-learn versions match, so no version-mismatch warning occurred.

## Implementation plan and findings

V5.3's Evil_Twin misses clustered by mission because its evidence was a modest relay-delay/AP-transition effect, while ordinary roaming included overlapping AP handoffs and management frames. Startup packet-age baselines were vulnerable to queue bursts. V5.4's first revision added `noninitial_association_duration_s` and association-relative latency, but large delivery gaps reset the whole extractor; the detector then forgot that its received AP had remained away from the initial AP. At high load, benign roaming could also hide the return-to-home packet and mimic a long association. Those were implementation/observability issues, not an effect of Replay source-segment settings on the Evil_Twin generator.

The V5.4R1 code keeps run-level initial AP and current-association state across delivery gaps while restarting only the rolling temporal window and packet-history buffers. It adds hard-negative training/validation coverage for high-load benign roam, longer roam intervals, and packet loss, then selects model and threshold using validation cohorts only. Feature names and model input columns remain the V5.4 schema (22 base / 88 temporal); no AP identifier, label, scenario, run ID, seed, split, attack schedule, or replay setting enters the matrix. The active runtime derives schedule labels only after predictions for evaluation.

Changed files are the new `simulation/research_v5_4r1/` runtime, protocol and experiment definitions, `run_project_v5_4r1.py`, and the root `run_project.py` selection. V4, V5.3, and the earlier V5.4 experiment artifacts remain untouched. A pre-edit backup is at `/tmp/ACPDAD_backups/v5_4r1_preverification_20261108T112028Z/`.

## Historical V5.3 failure and V5.4 development baseline

The preserved V5.3 final test recorded **73.77%** Evil_Twin recall and **97.96%** precision. Its replay-delay/source-segment changed condition recorded **68.24%** recall with only four Evil_Twin missions; one mission was at 8.3% recall. Replay delay/source-segment settings affected Replay-labelled runs, not Evil_Twin generator behavior. V5.3's Evil_Twin evidence was a modest relay delay plus an AP transition that overlapped ordinary handoffs. The association-duration feature measured time since the latest AP transition rather than persistence away from the run's initial AP. Startup packet-age excess used the first 10 observations: in a weak validation mission, startup packet age was around 260 ms while active-attack packet age was around 40 ms, making attack evidence negative; 68 of 97 scored attack observations were missed, mostly as Benign. V5.3's final test remains historical and was not rerun or used for selection.

The first V5.4 development candidate added association features but reset the association state across large delivery gaps. Benign roaming could also appear persistent when high loss hid the return packet. This table compares saved V5.4 and V5.4R1 **validation** metrics; it does not compare or tune on final-test results.

| Validation condition | V5.4 EV precision | V5.4 EV recall | V5.4 benign FPR | V5.4R1 EV precision | V5.4R1 EV recall | V5.4R1 benign FPR |
|---|---:|---:|---:|---:|---:|---:|
| mission_initial_state | 96.48% | 92.53% | 1.86% | 99.16% | 97.85% | 0.94% |
| wind_background_traffic | 94.88% | 73.21% | 5.15% | 99.29% | 97.79% | 1.10% |
| attack_timing_duration_intensity | 99.31% | 91.34% | 2.37% | 99.55% | 98.21% | 0.95% |
| replay_delay_source_segment | 88.50% | 81.61% | 5.01% | 99.52% | 98.12% | 1.54% |
| unseen_ap_benign_roaming | 90.90% | 81.95% | 3.66% | 95.20% | 99.30% | 1.70% |

## Validation-only selection

Selected model: **histgb_temporal**, threshold **0.21**, checkpoint SHA-256 `066229d9a46dc0bc506f88a706e15b9bc52ae289c6f1a7c1d076406a703209d1`. Model fit used only the training split. Candidate/threshold selection used the training validation split and five separate validation-condition datasets; the frozen final seeds were not generated at that point.

| Candidate | Selected EV threshold | Validation-cohort EV precision | EV recall | Benign FPR | Trainval macro-F1 |
|---|---:|---:|---:|---:|---:|
| extratrees_temporal | 0.39 | 99.015% | 97.278% | 2.357% | 95.448% |
| logistic_temporal | 0.05 | 79.882% | 99.530% | 22.894% | 80.902% |
| histgb_temporal | 0.21 | 98.707% | 98.079% | 1.338% | 95.785% |
| extratrees_raw | 0.55 | 99.129% | 94.321% | 3.247% | 91.042% |

Validation-only condition metrics at the frozen selected threshold:

- `mission_initial_state`: EV precision 99.16%, recall 97.85%, benign FPR 0.94%.
- `wind_background_traffic`: EV precision 99.29%, recall 97.79%, benign FPR 1.10%.
- `attack_timing_duration_intensity`: EV precision 99.55%, recall 98.21%, benign FPR 0.95%.
- `replay_delay_source_segment`: EV precision 99.52%, recall 98.12%, benign FPR 1.54%.
- `unseen_ap_benign_roaming`: EV precision 95.20%, recall 99.30%, benign FPR 1.70%.

## Frozen final-test protocol

The existing V5.3 final test remains historical and was not rerun or used for this work. V5.4R1's fresh test specification was frozen before generation: `9960ac85a973a2da29ef8a0d215ebc31a8f0a3f74fa925e896c028f5772cc87a`. The pre-generation lock is `bfddf35474cb6bf69d9644b2d9002fbe2f587151c1fc466d7cc361a1135b9113`. The pre-score manifest (dataset, run-manifest, config and checksum hashes) was written before model evaluation. The frozen checkpoint digest is `066229d9a46dc0bc506f88a706e15b9bc52ae289c6f1a7c1d076406a703209d1`.

There are 5 prespecified changed-condition suites, **20 complete runs per class per condition**, 500 simulated missions total, and 100 independent Evil_Twin missions. All use distinct run seeds and held-out AP pools. Replay delay/source-segment stress applies to Replay-labelled missions; Evil_Twin missions in that suite share the prespecified high-load network condition and are not compound Replay attacks. Every scheduled attack interval comes from the simulator schedule and labels were joined after model inference. Run-clustered intervals resample whole missions within condition × scheduled class, with 5,000 bootstrap replicates and seed 120261108.

| Condition | Master seed | Test AP pool base | Runs per class |
|---|---:|---:|---:|
| mission_initial_state | 61100000 | 28100000 | 20 |
| wind_background_traffic | 62100000 | 28200000 | 20 |
| attack_timing_duration_intensity | 63100000 | 28300000 | 20 |
| replay_delay_source_segment | 64100000 | 28400000 | 20 |
| unseen_ap_benign_roaming | 65100000 | 28500000 | 20 |

## Frozen test results

Row-level precision/recall/confusion metrics use observations for which the delivered packet stream produced a complete temporal feature window and the model emitted a prediction. Scheduled samples with no delivered/scorable observation cannot receive a class prediction and are excluded from the row-level matrix; every mission remains in the run-level table, and an Evil_Twin mission with no scored attack rows counts as a miss for mission coverage and any-alert detection.

When all scheduled active Evil_Twin time samples are instead counted—including those with no emitted prediction—the system-level recall is **70.033%** (8995/12844; run-clustered 95% CI **65.044%–74.529%**). This stricter denominator matters when interpreting the row-level score.

Pooled Evil_Twin confusion counts: **TP 8995, FN 167, FP 162**. Precision is **98.231%**; recall is **98.177%** with run-clustered 95% interval **97.734%–98.584%**. Run-clustered precision interval is **97.773%–98.622%**. Overall benign FPR is **1.664%** (run-clustered 95% interval **1.399%–1.948%**). 95/100 Evil_Twin missions reached 60% recall; missions with no scored attack rows count as misses for this mission-level coverage measure. The cluster-bootstrap interval for that mission fraction is **91.0%–99.0%**. 90/100 missions had at least 30 scored attack-positive observations; every mission, including unscored ones, remains listed in `per_mission_metrics.csv`.

The separate mission-level any-alert calculation detected 95/100 scheduled Evil_Twin attack missions (combined alert can be a class alert or physics residual); 62/100 also had a combined alert before the scheduled attack. This is an operational caveat: early false alerts are common even though row-level Evil_Twin precision and benign false-positive rate meet their pooled targets. Its run-clustered 95% intervals are 91.0%–99.0% detection and 53.0%–71.0% pre-existing alerts.

| Condition | EV TP/support | EV precision | EV recall | Clustered 95% recall interval | Benign FPR | Missions ≥60% recall | Missions with ≥30 EV rows | Any-alert mission detections | Pre-attack alert missions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mission_initial_state | 1997/2028 | 99.502% | 98.471% | 97.716%–99.240% | 1.287% | 20/20 | 19/20 | 20/20 | 9/20 |
| wind_background_traffic | 1277/1312 | 99.377% | 97.332% | 96.026%–98.443% | 2.528% | 16/20 | 16/20 | 16/20 | 15/20 |
| attack_timing_duration_intensity | 1832/1878 | 99.403% | 97.551% | 96.239%–98.761% | 1.500% | 20/20 | 18/20 | 20/20 | 12/20 |
| replay_delay_source_segment | 1769/1803 | 99.215% | 98.114% | 96.880%–98.972% | 1.443% | 19/20 | 18/20 | 19/20 | 11/20 |
| unseen_ap_benign_roaming | 2120/2141 | 94.685% | 99.019% | 98.489%–99.484% | 1.755% | 20/20 | 19/20 | 20/20 | 15/20 |

| Condition | Scheduled-active EV TP/support | Recall counting no-output samples as misses | Run-clustered 95% interval | Scheduled samples with no prediction |
|---|---:|---:|---:|---:|
| mission_initial_state | 1997/2510 | 79.562% | 72.511%–84.721% | 482 |
| wind_background_traffic | 1277/2467 | 51.763% | 38.266%–64.072% | 1155 |
| attack_timing_duration_intensity | 1832/2447 | 74.867% | 62.854%–83.995% | 569 |
| replay_delay_source_segment | 1769/2768 | 63.909% | 50.700%–75.404% | 965 |
| unseen_ap_benign_roaming | 2120/2652 | 79.940% | 71.339%–86.525% | 511 |

The pooled Evil_Twin precision target is met. The `unseen_ap_benign_roaming` subset is **94.685%** precision, so the 95% precision bar would not be met if interpreted separately for every condition.

Pooled five-class metrics: accuracy **96.744%**, majority-class baseline **68.293%** (`Benign`), macro-F1 **93.599%**, balanced accuracy **92.299%**.

| Scheduled class | Precision | Recall | F1 | Scored support |
|---|---:|---:|---:|---:|
| Benign | 96.987% | 98.336% | 97.657% | 73158 |
| DoS | 85.028% | 74.200% | 79.246% | 4531 |
| Replay | 99.047% | 95.666% | 97.327% | 9991 |
| Evil_Twin | 98.231% | 98.177% | 98.204% | 9162 |
| FDI | 96.005% | 95.118% | 95.559% | 10282 |

Classifier acceptance targets on emitted, scorable prediction rows (assessed on this single frozen simulated test pass):

- Pooled EV recall ≥80%: **MET** (98.18%).
- Every changed condition EV recall ≥70%: **MET** (weakest 97.33%).
- Pooled EV precision ≥95%: **MET** (98.23%).
- Pooled benign FPR ≤6.5%: **MET** (1.66%).
- At least 80% of EV missions at ≥60% recall: **MET** (95/100).

If every scheduled attack sample without a prediction counts as a miss, the pooled 80% system-level recall target is **NOT MET** (70.03%). The target that every changed condition reach 70% under that denominator is **NOT MET**: wind/background traffic is 51.76%, and replay-delay/source-segment is 63.91%. These system-level misses reflect packet delivery loss/delay and temporal warm-up/gaps: the classifier performs strongly on rows it receives, but high-load conditions do not consistently deliver enough scorable attack observations.

The behavioral distinction remains simulated: takeover produces a persistent reassociation to a per-run random AP identity with a relay path, while benign missions include random temporary roaming and management frames. The model receives only packet-derived transition persistence, packet age/delay, network summaries and physical telemetry; it receives neither AP identity nor attack state. Results support the five-class claim **within these tested synthetic conditions only**. The weakest non-EV results and all misses remain visible in the ledgers; DoS is the weakest pooled class in this test, so the overall five-class scores should not be read as equally strong class performance.

## Integrity and artifacts

All five dataset checksums match the pre-score lock; all 500 runs and seeds are unique; checkpoint, source, test specification and config hashes still match their pre-generation locks; each per-condition dashboard's local images resolve; confusion matrices were recalculated from saved predictions and match the runtime metrics. `integrity_checks` in `analysis/final_metrics.json` contains each condition's evidence.

- Prediction ledgers and dashboards: `evaluations/<condition>/`.
- Combined predictions, per-mission metrics, per-class/per-condition metrics, confusion matrices, figures and bootstrap intervals: `analysis/`.
- Pre-generation protocol lock: `pre_generation_lock.json`.
- Pre-score dataset/run-manifest hashes: `pre_score_manifest.json`.
- One-pass scoring record: `evaluation_execution.json`.
- Console logs: `../logs/`.

## Commands

From the project root, use `.venv`:

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib-cache"
.venv/bin/python run_project.py verify
.venv/bin/python run_project.py demo --data data/ACPDAD_Simulated_Research_v5_4r1_trainval --model simulation/models/acpdad_research_v5_4r1 --split val --output results/acpdad_research_v5_4r1/development/demo/manual
```

Fresh simulated generation and selected-run inference:

```bash
.venv/bin/python run_project.py generate --seed 9654321 --runs-per-class 1 --duration 30 --profile patrol --output data/acpdad_v5_4r1_new_seed_9654321
RUN_ID=$(.venv/bin/python -c "import pandas as pd; r=pd.read_csv('data/acpdad_v5_4r1_new_seed_9654321/run_manifest.csv'); print(r.loc[r.scenario.eq('Evil_Twin'),'run_id'].iloc[0])")
.venv/bin/python run_project.py predict --data data/acpdad_v5_4r1_new_seed_9654321 --model simulation/models/acpdad_research_v5_4r1 --run-id "$RUN_ID" --output results/acpdad_research_v5_4r1/prediction_checks/new_seed_9654321.csv
```

This is a simulation-only research workflow. It does not run SITL, connect to a vehicle, capture Wi-Fi, or establish real-world detection performance.
