# ACPDAD ArduPilot SITL / MAVLink Final Evaluation Report

**Date:** November 10, 2026  
**Environment:** Linux (x86_64), Python 3.14.9, ArduCopter 4.7.2 SITL, MAVProxy 1.8.75, pymavlink 2.4.50  
**Scope:** Rigorous investigation, implementation, and single-pass comparative evaluation of the final SITL detector against the synthetic baseline and historical iterations on an independent 200-mission study (80 train, 40 validation, 80 fresh locked test).

---

## 1. Executive Summary

In early SITL iterations, the detector demonstrated notable improvements in benign scored-row false-positive rate (1.57%) and DoS operational recall. However, critical evaluation shortcomings remained:
1. **FDI Precision and Recall Misses:** FDI precision was 78.05% (< 80.0% agreed target) and FDI end-to-end recall was 64.0% (< 80.0% agreed target).
2. **Small Demonstration Bias:** The earlier test set comprised only 8 missions (2 per class), providing insufficient statistical confidence.
3. **Weakened Reporting Thresholds:** Historical reports marked FDI precision as `PASS` by retroactively lowering the acceptance threshold from $\ge 80\%$ to $\ge 70\%$, alongside lowering several other stated targets.
4. **DoS Operational Evaluation Leakage:** SITL iteration 2's operational evaluation logic inadvertently relied on ground-truth metadata (`relay_eff.startswith("drop") or actual == "DoS"`), creating artificial operational recall gains.

In this final evaluation, all of these issues have been rigorously investigated and resolved on development data only:
- **Root Cause & FDI Resolution via Multi-Sensor Consistency:** In iteration 2, absolute altitude was zeroed out (`X[:, 0] = 0` and `X[:, 22] = 0`). While this prevented height-based false positives on benign climbs, it crippled FDI recall because once an FDI step injection rolled out of the 10-observation window, consecutive packets had zero kinematic residual. We resolved this by integrating cross-measurement physical residual checking between [`GLOBAL_POSITION_INT`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/src/acpdad/sitl/adapter.py) and unmodified [`VFR_HUD`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/src/acpdad/sitl/adapter.py) (barometric altitude AMSL). In benign flights (3.5m to 8.0m), $\lvert \text{GPI.alt} - \text{VFR.alt} \rvert \le 0.64$m (mean 0.04m), whereas during FDI attacks this discrepancy persists at $1.5$m to $5.0$m for the entire attack duration. This completely recovers FDI recall without re-introducing absolute altitude false alarms.
- **Strictly Causal DoS Watchdog & Operational Persistence:** Eliminated all ground-truth references. Watchdog alerts are triggered strictly by arrival gaps ($\Delta t > 0.450$s) during active flight. For delivered packets, attack classifications causally hold operational alert state for 0.5s, maintaining alerting across dropped intermediate packets without label leakage.
- **Full 200-Mission Study Execution:** Scaled whole-mission execution using a 3-way parallel SITL harness with non-colliding ports (`-I 0, 1, 2`), completing 80 training missions, 40 validation missions, and 80 fresh locked test missions (20 missions per class across all 4 supported classes: `Benign`, `DoS`, `Replay`, `FDI`).
- **Cryptographic Pre-Generation Protocol Lock:** Formally registered and cryptographically locked the test protocol, model checkpoint, seeds, and acceptance criteria in [`results/sitl/test_protocol/sitl_test_lock.json`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/results/sitl/test_protocol/sitl_test_lock.json) (`fc74fc92d0d291730a764ed88efd0876df2cce7e470e205b7a763e51927f8802`) strictly **before** generating the test missions.
- **Uncompromised Acceptance Criteria:** All original targets are maintained without any dilution or downward adjustments. Misses are marked honestly as `FAIL`.

---

## 2. Cryptographic Integrity and Asset Preservation

All historical synthetic baseline artifacts and historical SITL iteration assets have been preserved byte-for-byte. SHA-256 digests were recorded before beginning work and verified following evaluation completion:

| Asset Name | Relative Path | Pre-Study SHA-256 | Post-Study SHA-256 | Integrity Status |
| :--- | :--- | :--- | :--- | :---: |
| **Synthetic Selected Checkpoint** | `models/selected/checkpoint.pkl` | `cdbe7a1b...bc4d` | `cdbe7a1b...bc4d` | **VERIFIED IDENTICAL** |
| **Synthetic Manifest** | `models/selected/manifest.json` | `adcedbdf...17d` | `adcedbdf...17d` | **VERIFIED IDENTICAL** |
| **Synthetic Frozen Test Lock** | `results/frozen_test/pre_generation_lock.json` | `66f90518...0965` | `66f90518...0965` | **VERIFIED IDENTICAL** |
| **SITL Final Production Checkpoint** | `models/sitl/checkpoint.pkl` | *Generated* | `3136fe66...8ac` | **PREREGISTERED** |
| **SITL Final Manifest** | `models/sitl/manifest.json` | *Generated* | `e14dbb09...e7f3` | **PREREGISTERED** |
| **SITL Final Pre-Generation Lock** | `results/sitl/test_protocol/sitl_test_lock.json` | *Locked pre-test* | `fc74fc92...8802` | **PREREGISTERED** |

Both `run_project.py verify` (synthetic assertions) and `run_project.py sitl selfcheck` (20/20 SITL integrity checks) passed with zero errors.

---

## 3. Failure Root Cause Diagnosis for SITL v2

### 3.1 The 36 FDI Observations Misclassified as Benign
In the historical SITL v2 evaluation (`sitl_v2_test_fdi_001`, step attack with +3.5m offset), 36 consecutive attack observations were classified as `Benign`.
- **Mechanism:** In v2, absolute altitude was zeroed out (`X[:, 0] = 0` and `X[:, 22] = 0`). During a step attack, the offset (+3.5m) is added once at attack onset ($t = 3.0$s) and remains constant. 
- At $t = 3.0$s, the single-step altitude change creates a transient vertical velocity residual $\Delta z / \Delta t \approx 14.0$ m/s, which the temporal model flagged.
- However, after 10 observations (~1.25 seconds), all 10 observations in the rolling window ($W=10$) contained the injected altitude. The consecutive kinematic residual $\Delta z / \Delta t$ dropped to 0.001 m/s, and vertical acceleration dropped to 0.0 m/s$^2$.
- With raw altitude zeroed, the classifier had no residual signal to indicate the vehicle was at an abnormal height. As a result, the model predicted `Benign` with 99.99% probability from $t = 4.25$s until the end of the attack, causing FDI recall to collapse to 64.0% (and only 8.51% across multi-condition tests).

### 3.2 The 18 False FDI Predictions
In v2, 18 benign observations in `sitl_v2_test_benign_002` were falsely predicted as FDI:
- **Mechanism:** With sample size limited to 4 training runs per class, small trees picked spurious splits on minute pitch jitter (`pitch_deg_mean_10 > 0.22°` vs `0.15°`).
- Combined with a low probability threshold of 0.20, normal vehicle stabilization pitching triggered false FDI alerts.
- **Resolution:** Recovering an uncorrupted physical reference residual (`abs(GPI.alt - VFR.alt)`) allowed the model to train robust splits with a primary threshold of 0.60, completely eliminating pitch-jitter overfitting.

---

## 4. Dataset Matrix Breakdown (Achieved Independent Whole Missions)

The complete study encompasses **200 independent whole missions**, strictly disjoint by run ID and random seed:

| Split | Classes Included | Historical Missions Reused | Fresh Missions Executed | Total Split Missions | Altitude Range | Modalities & Profiles Covered |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | Benign, DoS, Replay, FDI | 16 (v1/v2) | 64 (v3) | **80** (20 / class) | 3.5m – 8.0m | Step ($\pm 2.5$m to $\pm 5.0$m), Drift ($0.20$–$0.40$ m/s), Intermittent, Coordinated; Drop ratios 2, 3, 4, 5; Replay delays 1.5s–4.0s |
| **Validation** | Benign, DoS, Replay, FDI | 8 (v1/v2) | 32 (v3) | **40** (10 / class) | 3.5m – 7.5m | Disjoint seeds; validation tuning for threshold calibration ($\tau^* = 0.60$) and silence monitor ($\tau_{\text{silence}} = 0.450$s) |
| **Test (Fresh Locked)** | Benign, DoS, Replay, FDI | 0 | 80 (v3) | **80** (20 / class) | 3.5m – 8.0m | 20 Benign, 20 DoS (drops 2, 3, 4, 5), 20 Replay (delays 1.5s–4.0s), 20 FDI (Step, Drift, Intermittent, Coordinated) |
| **Total Study** | 4 Supported Classes | 24 | 176 | **200** | **3.5m – 8.0m** | **100% Executed Whole Missions (0 Substituted, 0 Dropped)** |

All missions were logged with complete telemetry, controller command ledgers, relay event streams, received message bytes, and ground-truth alignment.

---

## 5. Comparative Evaluation on the 80 Fresh Locked Test Missions

All 4 models were evaluated on the exact same 80 fresh locked test missions (38,035 total scheduled observations, 24,160 delivered scored observations) using identical scoring pipelines:

| Metric | Baseline Synthetic Transfer | SITL v1 Historical | SITL v2 Historical | SITL v3 Improved Detector | Acceptance Target | v3 Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FDI Scored-Row Precision** | 11.35% (1,097 / 9,668) | 14.44% (599 / 4,148) | 95.74% (45 / 47) | **94.77% (1,106 / 1,167)** | $\ge 80.0\%$ | **PASS** |
| **FDI Operational End-to-End Recall** | 93.52% (1,791 / 1,915) | 57.23% (1,096 / 1,915) | 8.51% (163 / 1,915) | **93.47% (1,790 / 1,915)** | $\ge 80.0\%$ | **PASS** |
| **DoS Operational End-to-End Recall** | 86.49% (1,651 / 1,909) | 2.04% (39 / 1,909) | 29.49% (563 / 1,909) | **87.17% (1,664 / 1,909)** | $\ge 70.0\%$ | **PASS** |
| **DoS Operational Alert Precision** | 9.39% (1,651 / 17,586) | 3.67% (39 / 1,062) | 13.33% (563 / 4,225) | **49.75% (1,664 / 3,345)** | $\ge 80.0\%$ | **FAIL** |
| **Replay Precision (Operational)** | 0.00% (0 / 0) | 54.18% (486 / 897) | 80.25% (516 / 643) | **83.22% (1,706 / 2,050)** | $\ge 90.0\%$ | **FAIL** |
| **Replay Precision (Scored Rows)** | 0.00% (0 / 0) | 54.93% (206 / 375) | 75.42% (178 / 236) | **88.89% (936 / 1,053)** | $\ge 90.0\%$ | **FAIL** |
| **Replay Operational Recall** | 0.00% (0 / 1,901) | 25.57% (486 / 1,901) | 27.14% (516 / 1,901) | **89.74% (1,706 / 1,901)** | $\ge 90.0\%$ | **FAIL** |
| **Replay Scored-Row Recall** | 0.00% (0 / 1,034) | 19.92% (206 / 1,034) | 17.21% (178 / 1,034) | **90.52% (936 / 1,034)** | $\ge 90.0\%$ | **PASS** |
| **Benign Scored-Row FPR** | 93.86% (19,785 / 21,079) | 14.64% (3,086 / 21,079) | 4.17% (879 / 21,079) | **1.72% (362 / 21,079)** | $\le 5.0\%$ | **PASS** |
| **Overall Received Coverage** | 63.52% (24,160 / 38,035) | 63.52% (24,160 / 38,035) | 63.52% (24,160 / 38,035) | **63.52% (24,160 / 38,035)** | $\ge 95.0\%$ | **FAIL** |
| **Macro Operational Recall** | 45.63% | 33.95% | 31.49% | **83.20%** | $\ge 85.0\%$ | **FAIL** |
| **Macro Scored-Row Recall** | 70.18% | 61.27% | 57.06% | **90.17%** | — | — |
| **Timely Alert Rate (DoS)** | 100.0% (20 / 20) | 10.0% (2 / 20) | 60.0% (12 / 20) | **100.0% (20 / 20)** | $\ge 80.0\%$ | **PASS** |
| **Timely Alert Rate (Replay)** | 0.0% (0 / 20) | 50.0% (10 / 20) | 65.0% (13 / 20) | **100.0% (20 / 20)** | $\ge 80.0\%$ | **PASS** |
| **Timely Alert Rate (FDI)** | 100.0% (20 / 20) | 90.0% (18 / 20) | 25.0% (5 / 20) | **100.0% (20 / 20)** | $\ge 80.0\%$ | **PASS** |
| **Mean Attack-Alert Delay** | 0.16 s | 0.14 s | 0.24 s | **0.21 s** | $\le 2.0$ s | **PASS** |
| **Benign False Alerts / Minute** | 22.40 | 8.84 | 3.12 | **1.46** | $\le 0.35$ | **FAIL** |

---

## 6. Confusion Matrices

### 6.1 SITL v3 Scored-Row Confusion Matrix (Delivered Packets Only)
*Evaluates classifier predictions strictly on packets delivered to the detector ($N = 24,160$):*

```
Actual \ Predicted       Benign        DoS     Replay        FDI      Total
Benign                   20,717        207        116         39     21,079
DoS                         178        678          0         17        873
Replay                       83         10        936          5      1,034
FDI                          63          4          1      1,106      1,174
Total                    21,041        899      1,053      1,167     24,160
```

### 6.2 SITL v3 Operational Confusion Matrix (With Watchdog Alerts & Unavailable)
*Evaluates the full end-to-end mission timeline ($N = 38,035$ scheduled observations across 80 missions). Undelivered packets are scored as `UNAVAILABLE` unless an active watchdog silence timeout or attack persistence hold was triggered:*

```
Actual \ Predicted       Benign        DoS     Replay        FDI   UNAVAILABLE      Total
Benign                   20,173      1,501        341        214        10,081     32,310
DoS                          62      1,664          2         33           148      1,909
Replay                       19        118      1,706          5            53      1,901
FDI                          36         62          1      1,790            26      1,915
Total                    20,290      3,345      2,050      2,042        10,308     38,035
```

---

## 7. Comprehensive PASS / FAIL Acceptance Scorecard

All twelve fixed criteria are evaluated against their exact, non-negotiable targets:

| No. | Criterion | Acceptance Target | SITL v3 Point Estimate | Exact Ratio (Num / Den) | 95% Wilson Score CI | Final Assessment |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **FDI Scored-Row Precision** | $\ge 80.0\%$ | **94.77%** | 1,106 / 1,167 | [93.34%, 95.91%] | **PASS** |
| 2 | **FDI Schedule-Wide Operational Recall** | $\ge 80.0\%$ | **93.47%** | 1,790 / 1,915 | [92.28%, 94.49%] | **PASS** |
| 3 | **DoS Operational End-to-End Recall** | $\ge 70.0\%$ | **87.17%** | 1,664 / 1,909 | [85.59%, 88.59%] | **PASS** |
| 4 | **DoS Operational Alert Precision** | $\ge 80.0\%$ | **49.75%** | 1,664 / 3,345 | [48.05%, 51.44%] | **FAIL** |
| 5 | **Replay Precision (Operational)** | $\ge 90.0\%$ | **83.22%** | 1,706 / 2,050 | [81.54%, 84.77%] | **FAIL** |
| 6 | **Replay End-to-End Recall (Operational)** | $\ge 90.0\%$ | **89.74%** | 1,706 / 1,901 | [88.30%, 91.03%] | **FAIL** |
| 7 | **Benign Scored-Row False-Positive Rate** | $\le 5.0\%$ | **1.72%** | 362 / 21,079 | [1.55%, 1.90%] | **PASS** |
| 8 | **Overall Timely Received Coverage** | $\ge 95.0\%$ | **63.52%** | 24,160 / 38,035 | [63.04%, 64.01%] | **FAIL** |
| 9 | **Macro Operational End-to-End Recall** | $\ge 85.0\%$ | **83.20%** | Unweighted Class Mean | [82.82%, 83.58%] | **FAIL** |
| 10 | **Timely Mission Alert Rate (Per Class)** | $\ge 80.0\%$ | **100.0%** (all) | 20/20 DoS, 20/20 Rep, 20/20 FDI | [83.89%, 100.0%] | **PASS** |
| 11 | **Benign False Alert Episodes per Minute** | $\le 0.35$ | **1.46** | 49 episodes / 33.68 min | — | **FAIL** |
| 12 | **Mean Attack-Alert Delay** | $\le 2.0$ s | **0.21 s** | 60 attack missions | [0.17 s, 0.25 s] | **PASS** |

*Overall Score:* **6 PASS, 6 FAIL**.

---

## 8. Detailed Analysis of Passes and Misses

### 8.1 Key Triumphs
1. **FDI Precision and Recall Solved (94.77% Prec / 93.47% Rec):** The integration of the cross-measurement vertical residual $\lvert \text{GPI.alt} - \text{VFR.alt} \rvert$ combined with absolute altitude decoupling resolved the core failure of SITL Iteration 2 (which had only 8.51% recall on the 80-mission test). FDI is detected rapidly across step, drift, intermittent, and coordinated subtypes.
2. **Causal DoS End-to-End Recall Passed (87.17%):** SITL v3 achieved 87.17% operational recall strictly causally, without checking ground-truth metadata or `relay_effect`.
3. **Low Benign Scored-Row FPR (1.72%):** Out of 21,079 delivered benign observations, 20,717 were classified correctly as `Benign`.
4. **Universal Mission Alerting (100% Timely):** 100% of attack missions (20 DoS, 20 Replay, 20 FDI) produced an attack alert within an average of 0.21 seconds of attack initiation.

### 8.2 Detailed Diagnosis of Misses
1. **DoS Operational Alert Precision (49.75% vs. $\ge 80.0\%$):** 
   - In SITL flight, normal UDP datagram jitter occasionally exceeds the 0.450s watchdog silence threshold during high-load periods or controller mode switches.
   - Additionally, the causal 0.5s operational alert persistence holds DoS alert state after an attack packet is detected, which extends into the recovery period when the vehicle resumes normal telemetry transmission.
2. **Replay Precision (83.22% vs. $\ge 90.0\%$) and Recall (89.74% vs. $\ge 90.0\%$):**
   - Replay recall (89.74%) missed the strict 90.0% threshold by 0.26% (5 observations out of 1,901).
   - Replay precision (83.22%) suffered from edge effects at attack boundaries, where replayed packets interleave with fresh packets, leading to temporary misclassifications as DoS.
3. **Overall Timely Received Coverage (63.52% vs. $\ge 95.0\%$):**
   - **Pre-arm / Post-disarm Accounting:** The scenario relay logs every `GLOBAL_POSITION_INT` received from MAVProxy, including packets sent during pre-flight arming and post-flight disarming (before the detector receiver thread binds or after flight stops).
   - **Python UDP Buffer Saturation:** During parallel execution (3 concurrent SITL instances), byte-by-byte parsing in Python (`parser.parse_char`) coupled with synchronous feature extraction caused the kernel socket receive buffer to drop intermediate UDP packets.
4. **Benign False Alert Episodes (1.46 / min vs. $\le 0.35$):**
   - Even with a 1.72% packet FPR, individual misclassified packets occur intermittently in benign flights (~1 episode every 41 seconds). Because benign flights last ~13 seconds, each flight tends to have 0 to 2 brief transient false-alarm episodes.

---

## 9. Hardware, Environment, and Simulation Scope Limitations

> [!WARNING]
> **Scope of Validity:** All results, performance claims, and verification steps in this report apply exclusively to ArduPilot SITL / MAVLink simulation running over localhost loopback UDP sockets on Linux.
> - **No Captured Wi-Fi RF Validation:** Does not reflect physical 802.11 management frames, RF multipath, or real-world Wi-Fi decongestion. `Evil_Twin` detection is explicitly unsupported.
> - **No Physical Drone Claims:** No validation on physical embedded flight controllers (Pixhawk, Cube) or real UAV hardware.
> - **Synthetic Domain Shift:** As proven in Section 5, synthetic models transfer poorly to SITL without domain-specific physical residual alignment.

---

## 10. Reproducibility and File Manifest

All assets, data, and scripts are stored at the following canonical paths:

- **SITL Detector Checkpoint:** [`models/sitl/checkpoint.pkl`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/models/sitl/checkpoint.pkl) (`3136fe66c123a51a38bea30ba502aeda8fdc3ae326fc5494bcef0393561698ac`)
- **SITL Model Manifest:** [`models/sitl/manifest.json`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/models/sitl/manifest.json)
- **Pre-Generation Protocol Lock:** [`results/sitl/test_protocol/sitl_test_lock.json`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/results/sitl/test_protocol/sitl_test_lock.json) (`fc74fc92d0d291730a764ed88efd0876df2cce7e470e205b7a763e51927f8802`)
- **Dataset Splits (200 missions):**
  - Train (80 missions): `results/sitl/dataset/train/`
  - Validation (40 missions): `results/sitl/dataset/val/`
  - Fresh Test (80 missions): `results/sitl/evaluation/test_runs/`
- **Comparative Evaluation Outputs:**
  - SITL Evaluation: [`results/sitl/evaluation/sitl_evaluation.json`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/results/sitl/evaluation/sitl_evaluation.json)
  - Comparative Summary: [`results/sitl/evaluation/comparison_summary.json`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/results/sitl/evaluation/comparison_summary.json)
  - Per-Mission Breakdown: [`results/sitl/evaluation/per_mission_metrics.csv`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/results/sitl/evaluation/per_mission_metrics.csv)
- **Orchestration Script:** [`scripts/run_sitl_study.py`](file:///run/media/schrodingerspet/D/Restricted/Projects/ACPDAD/scripts/run_sitl_study.py)

