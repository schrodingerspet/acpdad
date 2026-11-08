# ACPDAD V5.4R2 focused system improvement

All records are simulated. This work does not establish ArduPilot SITL, physical-drone, captured-Wi-Fi, or real-world behavior.

## Diagnosis and implementation plan

V5.4R1 missed scheduled attack rows when packets were lost, exceeded the delivery-lag deadline, or the temporal window was warming back up after a gap. Its 3,682 unpredicted active Evil_Twin rows comprised 1,612 random losses, 1,386 queue/lag timeouts, and 684 post-gap warm-up rows. The prior alert policy also allowed physics and classifier evidence to combine into pre-attack alerts, so the 62/100 mission count could not be treated as Evil_Twin classifier errors alone.

The implementation plan kept the V5.4R1-compatible 22-base/88-temporal primary input and checkpoint contract, then used training/validation missions to select a shorter causal fallback, freshness-aware persistence, and a queue-priority assumption at the existing configured capacity and loss rates. The final test was specified and locked before generation and was scored once after freezing the model and policy. After that single scoring pass, I corrected a report-only denominator bug using saved prediction/audit rows; predictions, labels, and frozen test data were not rerun or changed.

The runtime and gap/alert policy are in `simulation/research_v5_4r2/experiment.py` and `core.py`; the final protocol, seed/config lock and aggregate reports are in `simulation/research_v5_4r2/protocol.py`. The corrected report-only calculation is `simulation/research_v5_4r2/reporting_correction.py`. The source digest used for frozen scoring remains in `pre_generation_lock.json`; post-score reporting source digests and generated-once artifact hashes are in `analysis/reporting_correction.json`.

## Frozen test and implementation

The fresh test was generated and evaluated once after freezing the V5.4R2 source, selected checkpoint, settings, seeds and conditions. Test specification SHA-256: `c1cca3555e2322685140b96362a6329d760b0e1c7737ab8a760f21bcd7cc8e70`. Pre-generation lock SHA-256: `66f905181503db16f8e6610d86d6e0b2c4820b71da6e1be276b8693edd000965`. Checkpoint SHA-256: `cdbe7a1b7fba8823c3c17ca722b34002c0852de446c5e7e0a41ba878a741bc4d`. The V5.4R1 frozen test remains untouched and historical.

The primary 22-base/88-temporal feature contract is unchanged. A separately trained three-observation fallback is used only until the primary window is ready after startup or a delivery gap. The validation-selected policy uses fresh K-of-N class evidence, benign hysteresis clearing, a bounded evidence expiry, separate physics warnings and separate communication-health warnings. Physics residuals alone no longer assert an attack. A telemetry expedited-priority queue reduces simulated queueing delay while leaving offered traffic, configured capacity, random loss, jitter, relay/handoff delay and the 450 ms maximum delivery lag unchanged. This is a modeled scheduling assumption, not an observed wireless guarantee.

Prediction deadline: 0.5 seconds after the current emitted observation time. A delayed offline reconstruction is retained in the ledger but is not counted as timely or allowed to keep an alert active.

## Gap accounting

V5.4R1 historical scheduled Evil_Twin gaps: 1,612 random network losses, 1,386 queue/lag timeouts, 684 post-gap temporal warm-up observations; initial startup accounted for none of the active attack rows. V5.4R1 pre-attack combined-alert sources: physics-only 47 missions, classifier-only 9, both 6. In the new test, 14/100 Evil_Twin missions had a pre-attack classifier alert; V5.4R2 physics-only warnings are explicitly separate and cannot assert an attack, so the 62-to-14 comparison is descriptive because the alert semantics changed.

V5.4R2 gap reasons are mutually exclusive per scheduled sample: `{"packet_never_delivered_network_loss": 11186}`. The total unavailable/late count is 11186; it reconciles against all 150000 scheduled timeline rows. Each observation audit records observation/arrival/prediction time, evidence age, delivery reason and timely/unavailable status. The complete breakdown is in `analysis/gap_breakdown.csv`.

## Results

Pooled prediction coverage: 92.54% (run-clustered 95% CI 92.22%–92.87%). Conditional Evil_Twin recall on timely scored rows: 98.75%. End-to-end Evil_Twin recall with every scheduled sample included: 84.76% (run-clustered 95% CI 83.90%–85.69%). Confusion counts: TP 10645, FN 1914, FP 207. Evil_Twin precision: 98.09% (95% CI 97.47%–98.57%). Benign scored-row FPR: 3.38% (95% CI 2.88%–3.93%).

| Condition | Coverage | Conditional EV recall | End-to-end EV recall (run-clustered 95% CI) | EV precision (run-clustered 95% CI) | Benign scored FPR | Benign mission alert rate | False episodes/min | EV pre-alert rate | EV alert delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mission_initial_state | 93.93% | 99.02% | 87.35% (86.09%–88.55%) | 99.67% (99.27%–99.88%) | 2.80% | 30.00% | 0.700 | 5.00% | 1.169s |
| wind_background_traffic | 90.88% | 98.16% | 82.11% (79.78%–84.45%) | 98.72% (97.47%–99.45%) | 4.07% | 65.00% | 1.200 | 20.00% | 1.374s |
| attack_timing_duration_intensity | 93.65% | 98.53% | 87.04% (85.43%–88.54%) | 99.50% (98.94%–99.83%) | 3.06% | 25.00% | 0.600 | 15.00% | 1.279s |
| replay_delay_source_segment | 90.94% | 99.03% | 82.11% (80.18%–84.05%) | 98.98% (97.87%–99.57%) | 4.36% | 40.00% | 1.000 | 10.00% | 1.209s |
| unseen_ap_benign_roaming | 93.32% | 99.01% | 85.37% (83.71%–87.02%) | 93.75% (89.82%–96.00%) | 2.66% | 20.00% | 0.450 | 20.00% | 1.137s |

## Frozen acceptance results

- PASS `pooled_end_to_end_evil_twin_recall`: measured `0.8475993311569392`; frozen target >= 80.0%.
- PASS `every_condition_end_to_end_evil_twin_recall`: measured `{"attack_timing_duration_intensity": 0.8704214257581725, "mission_initial_state": 0.8734593262119967, "replay_delay_source_segment": 0.8210850327048865, "unseen_ap_benign_roaming": 0.8536585365853658, "wind_background_traffic": 0.8211317768104471}`; frozen target >= 70.0% each.
- PASS `pooled_evil_twin_precision`: measured `0.980925175082934`; frozen target >= 95.0%.
- FAIL `every_condition_evil_twin_precision`: measured `{"attack_timing_duration_intensity": 0.995047276001801, "mission_initial_state": 0.9967182372245663, "replay_delay_source_segment": 0.9897959183673469, "unseen_ap_benign_roaming": 0.9375, "wind_background_traffic": 0.9871550903901046}`; frozen target >= 95.0% each.
- PASS `benign_scored_row_fpr`: measured `0.033815429983954204`; frozen target <= 6.5%.
- FAIL `benign_alert_mission_rate`: measured `0.36`; frozen target <= 20.0%.
- FAIL `false_alert_episode_rate`: measured `0.79`; frozen target <= 0.350/min.
- PASS `evil_twin_pre_attack_alert_rate`: measured `0.14`; frozen target <= 20.0%.
- PASS `mean_attack_alert_detection_delay`: measured `1.232989626105263`; frozen target <= 1.50s.

### Run-clustered alert-rate intervals

Intervals resample independent mission IDs (10,000 replicates), not correlated observations.

| Scope | Benign alert missions | False episodes / benign minute | Evil_Twin pre-attack alert missions |
|---|---:|---:|---:|
| pooled | 36/100 (36.0%; 95% CI 27.0%–46.0%) | 0.790 (95% CI 0.540–1.060) | 14/100 (14.0%; 95% CI 8.0%–21.0%) |
| mission_initial_state | 6/20 (30.0%; 95% CI 10.0%–50.0%) | 99.02% | 1/20 (5.0%; 95% CI 0.0%–15.0%) |
| wind_background_traffic | 13/20 (65.0%; 95% CI 45.0%–85.0%) | 98.16% | 4/20 (20.0%; 95% CI 5.0%–40.0%) |
| attack_timing_duration_intensity | 5/20 (25.0%; 95% CI 10.0%–45.0%) | 98.53% | 3/20 (15.0%; 95% CI 0.0%–30.0%) |
| replay_delay_source_segment | 8/20 (40.0%; 95% CI 20.0%–60.0%) | 99.03% | 2/20 (10.0%; 95% CI 0.0%–25.0%) |
| unseen_ap_benign_roaming | 4/20 (20.0%; 95% CI 5.0%–40.0%) | 99.01% | 4/20 (20.0%; 95% CI 5.0%–40.0%) |

### Five-class scored metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Benign | 90.86% | 96.62% | 0.937 | 92236 |
| DoS | 64.92% | 40.65% | 0.500 | 12019 |
| Replay | 98.65% | 95.25% | 0.969 | 12237 |
| Evil_Twin | 98.09% | 98.75% | 0.984 | 10780 |
| FDI | 98.33% | 89.80% | 0.939 | 11542 |

Scored-row accuracy 91.25%; macro-F1 0.866; balanced accuracy 0.842. Inference p50/p95/max latency: 14.16/16.70/153.57 ms under the recorded sequential streaming run.

### Alert source attribution

False classifier alert episode support by model path: `{"fallback_base": 1, "primary_temporal": 188}`. Pre-attack alert support by model path: `{"primary_temporal": 54}`. Physics warnings are logged separately and did not create classifier attack alerts; their episode count is 407 with support `{"fallback_base": 30, "primary_temporal": 380}`.


All targets met: **NO**. Failed targets remain reported as measured; no test conditions or predictions were changed.

## Alert and class details

Benign missions with any confirmed classifier attack alert: 36/100 (36.00%); false classifier alert episodes per benign mission-minute: 0.790; mean false episode duration: 3.663s. Evil_Twin missions with pre-attack alert: 14/100 (14.00%). Physics warning episodes remain separate and do not independently trigger attack classification. Communication warnings are separate from attack alerts.

100/100 Evil_Twin missions reached at least 60% schedule-wide recall; per-mission measurements for all classes and missions, including zero-prediction missions, are retained in `analysis/per_mission_metrics.csv`.

The scheduled-observation confusion matrix includes an `UNAVAILABLE` prediction column: `analysis/confusion_matrix_end_to_end.csv`. Scored-only confusion matrix: `analysis/confusion_matrix_scored.csv`. Per-mission recall, precision, coverage, missing-observation counts and alert measures are in `analysis/per_mission_metrics.csv`.

## Integrity, verification and limitations


**Reporting erratum (post-evaluation):** A report-arithmetic defect in `simulation/research_v5_4r2/experiment.py` reused a per-mission denominator when formatting pooled conditional Evil_Twin recall, printing TP counts as percentages. Conditional recall was recomputed from the saved run-mapped observation audits and cross-checked against the five-class recall. Per-mission scored-support values were also checked against the audit and matched the original run-level values. No model inference was rerun and no prediction, label, threshold, or test condition changed. Byte-identical generated-once report artifacts and their original hashes are preserved in `analysis/generated_once/` and `evaluation_execution.json`. Recreate the derived report values with `.venv/bin/python simulation/research_v5_4r2/reporting_correction.py`; it does not load the model or rerun inference.

Training/threshold/alert settings used only train and validation missions. The lock records source, config, model and manifest hashes. All five prespecified conditions and every mission remain in saved artifacts. The simulator is simplified; the priority policy is an assumption, not empirical radio scheduling evidence.

See `pre_generation_lock.json`, `pre_score_manifest.json`, `evaluation_execution.json`, `cross_verification.csv`, `verification/verify_1794150968915546792.json`, condition-level audits, label-free ledgers and dashboards. The denominator regression check is `../verification/reporting_denominator_regression_check.json`; correction execution log: `../logs/reporting_correction.log`.

Dashboard HTML and local image/link references were checked for the aggregate page and all five condition pages; every referenced file resolves and raster assets decode. A browser screenshot could not be produced in this restricted environment: Chromium failed at profile/crashpad setup, and Firefox exited without a screenshot (retry exit 139). Browser rendering remains unverified; see `analysis/dashboard_asset_check.json` and the console logs.
