"""End-to-end execution of SITL Iteration 3 (v3) study.

Executes:
1. Setup dataset directories (80 train, 40 val, 80 fresh test).
2. Copies verified historical v1/v2 dev runs.
3. Generates new train and val missions using parallel SITL workers.
4. Re-extracts causal features with enhanced cross-measurement & kinematic physical residuals.
5. Trains and calibrates dedicated SITL v3 model on train/val only.
6. Pre-registers and locks test protocol sitl_test_lock_v3.json BEFORE generating test data.
7. Generates 80 fresh locked test missions across 4 classes.
8. Single-pass comparative evaluation of Synthetic, SITL v1, SITL v2, and SITL v3 models.
9. Exports all metrics, audits, manifests, and reports.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymavlink import mavutil
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acpdad.simulation import BASE, TEMPORAL, W, MAX_GAP_NS, CLASSES
from acpdad.sitl.adapter import FeatureAdapter, sitl_schema
from acpdad.sitl.detector import TransferDetector
from acpdad.sitl.runner import run_session
from acpdad.sitl.watchdog import CausalTelemetryWatchdog

SITL_CLASSES = ["Benign", "DoS", "Replay", "FDI"]


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def reextract_run_features(run_dir: Path | str) -> pd.DataFrame:
    """Re-extract causal features from raw received_messages.jsonl using enhanced FeatureAdapter."""
    run_dir = Path(run_dir).resolve()
    cap_dir = run_dir / "capture" if (run_dir / "capture").is_dir() else run_dir
    msg_path = cap_dir / "received_messages.jsonl"
    feat_path = cap_dir / "features.csv"

    if not msg_path.is_file():
        raise FileNotFoundError(f"Missing received_messages.jsonl in {cap_dir}")

    mav = mavutil.mavlink.MAVLink(None)
    run_id = run_dir.name
    adapter = FeatureAdapter(run_id)

    rows = []
    with open(msg_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            raw = bytes.fromhex(m["raw_frame_hex"])
            for b in raw:
                msg = mav.parse_char(bytes([b]))
                if msg is not None:
                    _, row = adapter.consume(msg, m["arrival_time_ns"])
                    if row is not None:
                        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feature_columns = [
        "run_id", "observation_id", "arrival_time_ns", "source_time_ms", "sequence", "system_id", "component_id",
        "reset_reason", *BASE, "temporal_ready",
    ]
    for col in feature_columns:
        if col not in feat_df.columns:
            feat_df[col] = np.nan
    feat_df = feat_df.reindex(columns=feature_columns)
    feat_df.to_csv(feat_path, index=False)
    return feat_df


def run_single_mission_worker(spec: Dict[str, Any], output_root: Path, instance_id: int) -> Tuple[str, bool, float, str]:
    """Execute a single SITL mission on dedicated instance endpoints."""
    run_id = spec["run_id"]
    run_dir = output_root / run_id
    if (run_dir / "capture" / "features.csv").is_file() and (run_dir / "capture" / "ground_truth_joined_after_inference.csv").is_file():
        return run_id, True, 0.0, "reused"

    cfg_base = json.loads((ROOT / "config" / "sitl.json").read_text())
    port_offset = instance_id * 10
    cfg = json.loads(json.dumps(cfg_base))
    cfg["instance"] = instance_id
    cfg["endpoints"]["sitl_master"] = f"tcp:127.0.0.1:{5760 + port_offset}"
    cfg["endpoints"]["controller"] = f"udpin:127.0.0.1:{14550 + port_offset}"
    cfg["endpoints"]["controller_gcs_forward"] = f"udp:127.0.0.1:{14550 + port_offset}"
    cfg["endpoints"]["proxy_relay_forward"] = f"udp:127.0.0.1:{14552 + port_offset}"
    cfg["endpoints"]["relay_bind_port"] = 14552 + port_offset
    cfg["endpoints"]["detector_bind_port"] = 14551 + port_offset
    cfg["vehicle"]["speedup"] = 4

    tmp_cfg_file = Path(f"/tmp/sitl_v3_cfg_inst_{instance_id}_{os.getpid()}_{run_id}.json")
    tmp_cfg_file.write_text(json.dumps(cfg))

    start = time.time()
    last_err = ""
    for attempt in range(2):
        try:
            res = run_session(
                scenario=spec["scenario"],
                duration_s=spec.get("duration_s", 12.0),
                attack_start_s=spec.get("attack_start_s"),
                attack_duration_s=spec.get("attack_duration_s"),
                fdi_subtype=spec.get("fdi_subtype"),
                replay_delay_s=spec.get("replay_delay_s"),
                replay_repetitions=spec.get("replay_repetitions"),
                run_id=run_id,
                output_root=output_root,
                config_path=tmp_cfg_file,
                altitude_m=spec.get("altitude_m"),
                fdi_offset_m=spec.get("fdi_offset_m"),
                dos_drop_every=spec.get("dos_drop_every"),
            )
            elapsed = time.time() - start
            tmp_cfg_file.unlink(missing_ok=True)
            # Re-extract features with updated adapter
            reextract_run_features(run_dir)
            return run_id, True, elapsed, str(res.get("status"))
        except Exception as exc:
            last_err = str(exc)
            time.sleep(2.0)

    tmp_cfg_file.unlink(missing_ok=True)
    return run_id, False, time.time() - start, f"failed: {last_err}"


def execute_parallel_batch(specs: List[Dict[str, Any]], output_root: Path, num_workers: int = 3) -> List[Path]:
    """Execute a list of mission specs across parallel SITL workers."""
    output_root.mkdir(parents=True, exist_ok=True)
    completed = []
    to_run = [s for s in specs if not ((output_root / s["run_id"] / "capture" / "features.csv").is_file() and (output_root / s["run_id"] / "capture" / "ground_truth_joined_after_inference.csv").is_file())]
    reused = [s for s in specs if s not in to_run]

    for s in reused:
        run_d = output_root / s["run_id"]
        completed.append(run_d)

    print(f"Executing batch of {len(specs)} missions ({len(reused)} existing, {len(to_run)} to run) across {num_workers} parallel workers...")
    if not to_run:
        return completed

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Distribute missions across available instance IDs (0..num_workers-1)
        future_map = {}
        for idx, spec in enumerate(to_run):
            inst_id = idx % num_workers
            fut = executor.submit(run_single_mission_worker, spec, output_root, inst_id)
            future_map[fut] = spec

        done_count = len(reused)
        total_count = len(specs)
        for fut in concurrent.futures.as_completed(future_map):
            run_id, success, elapsed, status = fut.result()
            done_count += 1
            if success:
                print(f"  [{done_count}/{total_count}] Mission {run_id} ({status}) in {elapsed:.1f}s")
                completed.append(output_root / run_id)
            else:
                print(f"  [{done_count}/{total_count}] Mission {run_id} FAILED: {status}")
                raise RuntimeError(f"Mission {run_id} failed: {status}")

    return sorted(completed, key=lambda p: p.name)


def generate_mission_specs(split_name: str, runs_per_class: int, id_offset: int = 1) -> List[Dict[str, Any]]:
    """Deterministically generate diverse mission specifications."""
    specs = []
    altitudes = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    fdi_offsets = [4.0, 3.5, 3.0, 2.5, -2.5, -3.0, -3.5, -4.0, 5.0, -5.0]
    drift_rates = [0.20, 0.25, 0.30, 0.35, 0.40]
    dos_drops = [2, 3, 4, 5, 2, 3, 2, 4]
    replay_delays = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 2.0, 3.0]

    for idx in range(runs_per_class):
        run_num = id_offset + idx
        alt = altitudes[idx % len(altitudes)]
        dur = 12.0 + (idx % 3) * 1.0  # 12.0, 13.0, 14.0

        # Benign
        specs.append({
            "run_id": f"sitl_{split_name}_benign_{run_num:03d}",
            "scenario": "benign",
            "duration_s": dur,
            "altitude_m": alt,
        })

        # FDI
        fdi_type = ["step", "drift", "intermittent", "coordinated"][idx % 4]
        fdi_spec = {
            "run_id": f"sitl_{split_name}_fdi_{run_num:03d}",
            "scenario": "FDI",
            "duration_s": dur,
            "altitude_m": alt,
            "attack_start_s": 3.0,
            "attack_duration_s": 6.0,
            "fdi_subtype": fdi_type,
            "fdi_offset_m": fdi_offsets[idx % len(fdi_offsets)],
        }
        specs.append(fdi_spec)

        # DoS
        specs.append({
            "run_id": f"sitl_{split_name}_dos_{run_num:03d}",
            "scenario": "DoS",
            "duration_s": dur,
            "altitude_m": alt,
            "attack_start_s": 3.0,
            "attack_duration_s": 6.0,
            "dos_drop_every": dos_drops[idx % len(dos_drops)],
        })

        # Replay
        specs.append({
            "run_id": f"sitl_{split_name}_replay_{run_num:03d}",
            "scenario": "Replay",
            "duration_s": dur,
            "altitude_m": alt,
            "attack_start_s": 3.0,
            "attack_duration_s": 6.0,
            "replay_delay_s": replay_delays[idx % len(replay_delays)],
            "replay_repetitions": 1 if idx % 2 == 0 else 2,
        })

    return specs


def extract_features_and_labels_v3(run_dirs: List[Path], neutralize_altitude: bool = True) -> Dict[str, Any]:
    """Causally extract temporal features from run features.csv."""
    all_base = []
    all_temporal = []
    all_labels = []
    all_runs = []

    for r_dir in run_dirs:
        feat_df = pd.read_csv(r_dir / "capture" / "features.csv")
        gt_df = pd.read_csv(r_dir / "capture" / "ground_truth_joined_after_inference.csv")
        cov = gt_df[gt_df["coverage"]]
        min_len = min(len(feat_df), len(cov))
        feat_df = feat_df.iloc[:min_len]
        labels = cov["Target_Label"].values[:min_len]

        base_vals = feat_df[list(BASE)].to_numpy(dtype=float)
        temporal_vals = np.full((len(feat_df), len(TEMPORAL)), np.nan, dtype=float)

        for i in range(len(feat_df)):
            window_slice = base_vals[max(0, i - W + 1): i + 1]
            raw = base_vals[i]
            mean = np.mean(window_slice, axis=0)
            std = np.std(window_slice, axis=0, ddof=1) if len(window_slice) > 1 else np.zeros_like(raw)
            diff = base_vals[i] - base_vals[i - 1] if i > 0 else np.zeros_like(raw)
            temporal_vals[i] = np.concatenate([raw, mean, std, diff])

        all_base.append(base_vals)
        all_temporal.append(temporal_vals)
        all_labels.extend(labels)
        all_runs.extend([r_dir.name] * len(feat_df))

    X_base = np.vstack(all_base) if all_base else np.empty((0, len(BASE)))
    X_temporal = np.vstack(all_temporal) if all_temporal else np.empty((0, len(TEMPORAL)))

    if neutralize_altitude:
        if X_base.shape[1] > 0:
            X_base[:, 0] = 0.0
        if X_temporal.shape[1] > 22:
            X_temporal[:, 0] = 0.0
            X_temporal[:, 22] = 0.0

    return {
        "X_base": X_base,
        "X_temporal": X_temporal,
        "y": np.array(all_labels),
        "run_ids": np.array(all_runs),
    }


def train_sitl_classifier_v3(
    train_dirs: List[Path],
    val_dirs: List[Path],
    output_model_dir: Path,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train HistGradientBoosting primary and fallback models with altitude decoupling."""
    train_data = extract_features_and_labels_v3(train_dirs, neutralize_altitude=True)
    val_data = extract_features_and_labels_v3(val_dirs, neutralize_altitude=True)

    X_train = np.nan_to_num(train_data["X_temporal"], nan=0.0)
    X_train_base = np.nan_to_num(train_data["X_base"], nan=0.0)
    y_train = train_data["y"]
    mask_train = np.isin(y_train, SITL_CLASSES)
    X_train = X_train[mask_train]
    X_train_base = X_train_base[mask_train]
    y_train = y_train[mask_train]

    X_val = np.nan_to_num(val_data["X_temporal"], nan=0.0)
    X_val_base = np.nan_to_num(val_data["X_base"], nan=0.0)
    y_val = val_data["y"]
    mask_val = np.isin(y_val, SITL_CLASSES)
    X_val = X_val[mask_val]
    X_val_base = X_val_base[mask_val]
    y_val = y_val[mask_val]

    primary_clf = HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=100,
        min_samples_leaf=15,
        class_weight="balanced",
    )
    primary_clf.fit(X_train, y_train)

    fallback_clf = HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=100,
        min_samples_leaf=15,
        class_weight="balanced",
    )
    fallback_clf.fit(X_train_base, y_train)

    # Calibrate operating threshold on validation set
    val_probs = primary_clf.predict_proba(X_val)
    classes = list(primary_clf.classes_)
    fdi_idx = classes.index("FDI")
    benign_idx = classes.index("Benign")

    best_th = 0.5
    best_score = -1.0
    val_metrics = {}

    for th in np.arange(0.30, 0.75, 0.05):
        preds = []
        for p in val_probs:
            max_c = classes[int(np.argmax(p))]
            # Apply calibrated threshold to FDI
            if p[fdi_idx] >= th:
                preds.append("FDI")
            elif max_c == "FDI":
                # Fallback to next best class
                p_copy = p.copy()
                p_copy[fdi_idx] = -1.0
                preds.append(classes[int(np.argmax(p_copy))])
            else:
                preds.append(max_c)

        fdi_prec = precision_score(y_val, preds, labels=["FDI"], average=None)[0]
        fdi_rec = recall_score(y_val, preds, labels=["FDI"], average=None)[0]
        benign_fp = sum((y_val == "Benign") & (np.array(preds) != "Benign"))
        benign_fpr = benign_fp / sum(y_val == "Benign")

        # Criterion: FDI precision >= 80%, FDI recall >= 80%, benign FPR <= 5%
        score = f1_score(y_val, preds, average="macro")
        if fdi_prec >= 0.80 and fdi_rec >= 0.80 and benign_fpr <= 0.05:
            score += 10.0  # Reward satisfying all strict criteria

        if score > best_score:
            best_score = score
            best_th = float(th)
            val_metrics = {
                "threshold": float(th),
                "fdi_precision": float(fdi_prec),
                "fdi_recall": float(fdi_rec),
                "benign_fpr": float(benign_fpr),
                "macro_f1": float(f1_score(y_val, preds, average="macro")),
            }

    print(f"Calibrated primary threshold: {best_th:.2f} on validation set: {val_metrics}")

    checkpoint = {
        "contract": "acpdad-sitl-v3",
        "provenance": "SITL_TRAINED_MODEL",
        "altitude_neutralized": True,
        "classes": SITL_CLASSES,
        "models": {
            "primary": primary_clf,
            "fallback_base": fallback_clf,
        },
        "primary_threshold": best_th,
        "fallback_threshold": 0.5,
        "fallback_selected": "base",
        "window": W,
        "base_features": list(BASE),
        "temporal_features": list(TEMPORAL),
        "physics_threshold": 0.0108,
        "training_runs": [p.name for p in train_dirs],
        "validation_runs": [p.name for p in val_dirs],
        "trained_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "Dedicated SITL v3 model with cross-measurement consistency, decoupled altitude, and causal watchdog.",
    }

    output_model_dir.mkdir(parents=True, exist_ok=True)
    ck_path = output_model_dir / "checkpoint.pkl"
    with open(ck_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    ck_sha = sha256_file(ck_path)
    manifest = {
        "schema_version": "acpdad-sitl-model-v3",
        "contract": "acpdad-sitl-v3",
        "provenance": "SITL_TRAINED_MODEL",
        "altitude_neutralized": True,
        "checkpoint_sha256": ck_sha,
        "checkpoint_file": "checkpoint.pkl",
        "classes": SITL_CLASSES,
        "feature_count_base": len(BASE),
        "feature_count_temporal": len(TEMPORAL),
        "primary_threshold": best_th,
        "fallback_threshold": 0.5,
        "training_runs_count": len(train_dirs),
        "validation_runs_count": len(val_dirs),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "checkpoint_sha256": ck_sha,
        "manifest": manifest,
        "val_metrics": val_metrics,
    }


def evaluate_run_strictly_causal(
    run_dir: Path,
    detector: TransferDetector,
    is_sitl_model: bool = True,
    silence_threshold_s: float = 0.450,
) -> Dict[str, Any]:
    """Causal evaluation strictly separating scored-row classifications, unavailable packets, and watchdog alerts."""
    feat_df = pd.read_csv(run_dir / "capture" / "features.csv")
    gt_df = pd.read_csv(run_dir / "capture" / "ground_truth_joined_after_inference.csv")
    run_id = run_dir.name

    scheduled_total = len(gt_df)
    adapter = FeatureAdapter(run_id)
    adapter.temporal_window = deque(maxlen=W)
    last_arrival = None

    predictions = []
    for r_dict in feat_df.to_dict("records"):
        arrival = int(r_dict["arrival_time_ns"])
        if last_arrival is not None and arrival - last_arrival > MAX_GAP_NS:
            adapter.temporal_window.clear()
            r_dict["reset_reason"] = "delivery_gap"
        else:
            r_dict["reset_reason"] = ""
        adapter.temporal_window.append(np.asarray([r_dict[name] for name in BASE], dtype=float))
        r_dict["temporal_ready"] = len(adapter.temporal_window) == W
        pred = detector.predict(adapter, r_dict)
        predictions.append(pred)
        last_arrival = arrival

    pred_iter = iter(predictions)
    timeline = []
    last_delivered_elapsed = None
    last_alert_time = None
    active_alert_type = None
    watchdog_timeout_alerts = 0
    benign_false_alert_episodes = 0
    in_benign_alert_episode = False

    for _, gt_row in gt_df.iterrows():
        ev_idx = int(gt_row["event_index"])
        actual = str(gt_row["Target_Label"])
        elapsed = float(gt_row["mission_elapsed_s"]) if pd.notna(gt_row.get("mission_elapsed_s")) else None
        is_cov = bool(gt_row.get("coverage", True))

        if is_cov:
            try:
                assigned = next(pred_iter)
                predicted = str(assigned["prediction"])
                conf = float(assigned.get("confidence", 0.0))
                covered = True
                last_delivered_elapsed = elapsed
                watchdog_alert = False
                if predicted != "Benign":
                    active_alert_type = predicted
                    last_alert_time = elapsed
                elif last_alert_time is not None and elapsed - last_alert_time > 0.5:
                    active_alert_type = None
            except StopIteration:
                predicted = "UNAVAILABLE"
                conf = 0.0
                covered = False
                watchdog_alert = False
        else:
            predicted = "UNAVAILABLE"
            conf = 0.0
            covered = False
            # Causal silence check: gap since last delivered packet
            gap_s = (elapsed - last_delivered_elapsed) if (elapsed is not None and last_delivered_elapsed is not None) else 0.0
            watchdog_alert = gap_s > silence_threshold_s
            if watchdog_alert:
                watchdog_timeout_alerts += 1

        # Operational alert status at this observation
        if covered:
            op_predicted = active_alert_type if active_alert_type is not None else predicted
        else:
            if watchdog_alert:
                op_predicted = "DoS"
            elif active_alert_type is not None and last_alert_time is not None and (elapsed - last_alert_time <= 0.5):
                op_predicted = active_alert_type
            else:
                op_predicted = "UNAVAILABLE"

        # Track benign false alert episodes
        if actual == "Benign":
            is_alert = (predicted != "Benign" and covered) or (op_predicted != "Benign" and op_predicted != "UNAVAILABLE")
            if is_alert and not in_benign_alert_episode:
                benign_false_alert_episodes += 1
                in_benign_alert_episode = True
            elif not is_alert:
                in_benign_alert_episode = False

        timeline.append({
            "observation_id": ev_idx,
            "mission_elapsed_s": elapsed,
            "actual": actual,
            "predicted_scored": predicted,
            "predicted_operational": op_predicted,
            "confidence": conf,
            "covered": covered,
            "watchdog_timeout_alert": watchdog_alert,
        })

    t_df = pd.DataFrame(timeline)
    covered_count = int(t_df["covered"].sum())
    coverage_rate = covered_count / scheduled_total if scheduled_total > 0 else 0.0

    # Scored rows confusion (Delivered packets only)
    scored_rows = t_df[t_df["covered"]]
    scored_conf = {a: {p: 0 for p in SITL_CLASSES} for a in SITL_CLASSES}
    for _, row in scored_rows.iterrows():
        a, p = row["actual"], row["predicted_scored"]
        if a in scored_conf and p in scored_conf[a]:
            scored_conf[a][p] += 1

    # Operational confusion (Delivered + Watchdog timeout alerts, with UNAVAILABLE)
    op_conf = {a: {p: 0 for p in SITL_CLASSES + ["UNAVAILABLE"]} for a in SITL_CLASSES}
    for _, row in t_df.iterrows():
        a, op = row["actual"], row["predicted_operational"]
        if a in op_conf and op in op_conf[a]:
            op_conf[a][op] += 1
        elif a in op_conf:
            op_conf[a]["UNAVAILABLE"] += 1

    # Detection latency
    attack_rows = t_df[t_df["actual"] != "Benign"]
    first_attack_time = attack_rows["mission_elapsed_s"].min() if len(attack_rows) > 0 else None
    detected = attack_rows[(attack_rows["predicted_scored"] == attack_rows["actual"]) | (attack_rows["predicted_operational"] == attack_rows["actual"])]
    first_detect_time = detected["mission_elapsed_s"].min() if len(detected) > 0 else None
    latency_s = (first_detect_time - first_attack_time) if (first_detect_time is not None and first_attack_time is not None) else None

    # Benign duration in minutes
    benign_rows = t_df[t_df["actual"] == "Benign"]
    benign_minutes = (len(benign_rows) * 0.25) / 60.0

    # Per-mission timely alert (alert within 2.0s of attack onset)
    produced_timely_alert = False
    if len(attack_rows) > 0 and latency_s is not None and latency_s <= 2.0:
        produced_timely_alert = True

    return {
        "run_id": run_id,
        "scheduled_observations": scheduled_total,
        "covered_observations": covered_count,
        "coverage_rate": coverage_rate,
        "scored_confusion": scored_conf,
        "operational_confusion": op_conf,
        "watchdog_timeout_alerts": watchdog_timeout_alerts,
        "benign_false_alert_episodes": benign_false_alert_episodes,
        "benign_minutes": benign_minutes,
        "detection_latency_s": latency_s,
        "produced_timely_alert": produced_timely_alert,
        "timeline": timeline,
    }


def aggregate_study_metrics(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-run metrics and compute Wilson score confidence intervals."""
    total_scheduled = sum(r["scheduled_observations"] for r in run_results)
    total_covered = sum(r["covered_observations"] for r in run_results)
    coverage_rate = total_covered / total_scheduled if total_scheduled > 0 else 0.0

    # Aggregate confusion matrices
    scored_conf = {a: {p: 0 for p in SITL_CLASSES} for a in SITL_CLASSES}
    op_conf = {a: {p: 0 for p in SITL_CLASSES + ["UNAVAILABLE"]} for a in SITL_CLASSES}

    for r in run_results:
        for a in SITL_CLASSES:
            for p in SITL_CLASSES:
                scored_conf[a][p] += r["scored_confusion"][a][p]
            for p in SITL_CLASSES + ["UNAVAILABLE"]:
                op_conf[a][p] += r["operational_confusion"][a][p]

    def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        if n == 0:
            return 0.0, 0.0
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    per_class = {}
    for c in SITL_CLASSES:
        # Scored row metrics
        scored_support = sum(scored_conf[c].values())
        scored_tp = scored_conf[c][c]
        scored_pred = sum(scored_conf[a][c] for a in SITL_CLASSES)
        scored_prec = scored_tp / scored_pred if scored_pred > 0 else 0.0
        scored_rec = scored_tp / scored_support if scored_support > 0 else 0.0
        prec_low, prec_high = wilson_ci(scored_tp, scored_pred)
        rec_low, rec_high = wilson_ci(scored_tp, scored_support)

        # Operational metrics (including unavailable)
        op_support = sum(op_conf[c].values())
        op_tp = op_conf[c][c]
        op_pred = sum(op_conf[a][c] for a in SITL_CLASSES)
        op_unavail = op_conf[c].get("UNAVAILABLE", 0)
        op_rec = op_tp / op_support if op_support > 0 else 0.0
        op_prec = op_tp / op_pred if op_pred > 0 else 0.0
        op_prec_low, op_prec_high = wilson_ci(op_tp, op_pred)
        op_rec_low, op_rec_high = wilson_ci(op_tp, op_support)

        per_class[c] = {
            "scored_support": scored_support,
            "scored_tp": scored_tp,
            "scored_predicted": scored_pred,
            "scored_precision": float(scored_prec),
            "scored_precision_ci95": [float(prec_low), float(prec_high)],
            "scored_recall": float(scored_rec),
            "scored_recall_ci95": [float(rec_low), float(rec_high)],
            "operational_support": op_support,
            "operational_tp": op_tp,
            "operational_predicted": op_pred,
            "operational_unavailable": op_unavail,
            "operational_precision": float(op_prec),
            "operational_precision_ci95": [float(op_prec_low), float(op_prec_high)],
            "operational_recall": float(op_rec),
            "operational_recall_ci95": [float(op_rec_low), float(op_rec_high)],
            "operational_f1": float(2 * op_prec * op_rec / (op_prec + op_rec)) if (op_prec + op_rec) > 0 else 0.0,
        }

    # Benign scored row FPR
    benign_scored_total = sum(scored_conf["Benign"].values())
    benign_fp = benign_scored_total - scored_conf["Benign"]["Benign"]
    benign_fpr = benign_fp / benign_scored_total if benign_scored_total > 0 else 0.0
    fpr_low, fpr_high = wilson_ci(benign_fp, benign_scored_total)

    # Benign false alert episodes per minute
    total_benign_episodes = sum(r["benign_false_alert_episodes"] for r in run_results)
    total_benign_minutes = sum(r["benign_minutes"] for r in run_results)
    episodes_per_minute = total_benign_episodes / total_benign_minutes if total_benign_minutes > 0 else 0.0

    # Latency across attack runs
    latencies = [r["detection_latency_s"] for r in run_results if r.get("detection_latency_s") is not None]
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    # Timely alert rate per attack class
    timely_alert_rate = {}
    for c in ["DoS", "Replay", "FDI"]:
        c_runs = [r for r in run_results if f"_{c.lower()}_" in r["run_id"]]
        timely_count = sum(1 for r in c_runs if r["produced_timely_alert"])
        timely_alert_rate[c] = {
            "missions": len(c_runs),
            "timely_alert_missions": timely_count,
            "rate": timely_count / len(c_runs) if c_runs else 0.0,
        }

    macro_op_recall = float(np.mean([per_class[c]["operational_recall"] for c in SITL_CLASSES]))

    return {
        "runs_evaluated": len(run_results),
        "total_scheduled_observations": total_scheduled,
        "total_covered_observations": total_covered,
        "overall_coverage_rate": float(coverage_rate),
        "scored_confusion_matrix": scored_conf,
        "operational_confusion_matrix": op_conf,
        "per_class": per_class,
        "benign_scored_fpr": float(benign_fpr),
        "benign_scored_fpr_ci95": [float(fpr_low), float(fpr_high)],
        "benign_false_alert_episodes_per_minute": float(episodes_per_minute),
        "average_detection_latency_s": avg_latency,
        "macro_operational_recall": macro_op_recall,
        "timely_alert_rate_by_class": timely_alert_rate,
    }


def main():
    print("=" * 70)
    print("=== ACPDAD ARDUPILOT SITL BENCHMARK & REPRODUCTION STUDY ===")
    print("=" * 70)

    sitl_root = ROOT / "results" / "sitl"
    train_root = sitl_root / "dataset" / "train"
    val_root = sitl_root / "dataset" / "val"
    test_root = sitl_root / "evaluation" / "test_runs"
    model_root = ROOT / "models" / "sitl"
    lock_path = sitl_root / "test_protocol" / "sitl_test_lock.json"
    eval_output_dir = sitl_root / "evaluation"

    # Step 1: Copy verified dev runs from v1/v2 to preserve continuity
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    v2_train = ROOT / "results" / "sitl_v2" / "dataset" / "train"
    if v2_train.is_dir():
        for r_dir in v2_train.iterdir():
            if r_dir.is_dir() and not (train_root / r_dir.name).is_dir():
                print(f"Copying verified training run {r_dir.name} -> {train_root.name}")
                shutil.copytree(r_dir, train_root / r_dir.name)

    v2_val = ROOT / "results" / "sitl_v2" / "dataset" / "val"
    if v2_val.is_dir():
        for r_dir in v2_val.iterdir():
            if r_dir.is_dir() and not (val_root / r_dir.name).is_dir():
                print(f"Copying verified validation run {r_dir.name} -> {val_root.name}")
                shutil.copytree(r_dir, val_root / r_dir.name)

    # Step 2: Define and execute remaining train missions (reach 20 per class = 80 total)
    # We have 4 per class existing (16 total). We need 16 per class = 64 new runs (ids 005 to 020).
    new_train_specs = generate_mission_specs("v3_train", runs_per_class=16, id_offset=5)
    print(f"\n=== [1/6] Executing {len(new_train_specs)} New Training Missions (Parallel 3 Workers) ===")
    execute_parallel_batch(new_train_specs, train_root, num_workers=3)

    # Step 3: Define and execute remaining val missions (reach 10 per class = 40 total)
    # We have 2 per class existing (8 total). We need 8 per class = 32 new runs (ids 003 to 010).
    new_val_specs = generate_mission_specs("v3_val", runs_per_class=8, id_offset=3)
    print(f"\n=== [2/6] Executing {len(new_val_specs)} New Validation Missions (Parallel 3 Workers) ===")
    execute_parallel_batch(new_val_specs, val_root, num_workers=3)

    # Re-extract features across ALL 80 train and 40 val runs with updated adapter
    train_dirs = sorted([d for d in train_root.iterdir() if d.is_dir()])
    val_dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])
    print(f"\nRe-extracting causal features with enhanced residual across {len(train_dirs)} train and {len(val_dirs)} val missions...")
    for d in train_dirs:
        reextract_run_features(d)
    for d in val_dirs:
        reextract_run_features(d)

    # Step 4: Train and Calibrate Model models/sitl_v3
    print(f"\n=== [3/6] Training and Calibrating Dedicated SITL v3 Model ===")
    train_res = train_sitl_classifier_v3(train_dirs, val_dirs, model_root, random_state=42)
    print(f"Model saved to {model_root} (checkpoint SHA-256: {train_res['checkpoint_sha256']})")

    # Step 5: Finalize and Pre-register Locked Test Protocol BEFORE generating test missions
    print(f"\n=== [4/6] Pre-registering and Locking SITL v3 Test Protocol ===")
    # 20 NEW whole missions per class = 80 fresh whole missions (ids 001 to 020)
    test_specs = generate_mission_specs("v3_test", runs_per_class=20, id_offset=1)

    source_files = [
        "src/acpdad/sitl/adapter.py", "src/acpdad/sitl/capture.py", "src/acpdad/sitl/cli.py",
        "src/acpdad/sitl/controller.py", "src/acpdad/sitl/detector.py", "src/acpdad/sitl/runner.py",
        "src/acpdad/sitl/scenarios.py", "src/acpdad/sitl/verification.py", "src/acpdad/sitl/watchdog.py",
        "src/acpdad/sitl/workflow.py",
    ]
    src_hashes = {Path(sf).name: sha256_file(ROOT / sf) for sf in source_files}

    train_records = [{"run_id": td.name, "features_sha256": sha256_file(td / "capture" / "features.csv"), "ground_truth_sha256": sha256_file(td / "capture" / "ground_truth_joined_after_inference.csv")} for td in train_dirs]
    val_records = [{"run_id": vd.name, "features_sha256": sha256_file(vd / "capture" / "features.csv"), "ground_truth_sha256": sha256_file(vd / "capture" / "ground_truth_joined_after_inference.csv")} for vd in val_dirs]

    lock_payload = {
        "schema_version": "acpdad-sitl-test-lock-v3",
        "contract": "acpdad-sitl-test-protocol-v3",
        "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
        "locked_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_hashes": src_hashes,
        "training_set": {"mission_count": len(train_records), "runs": train_records},
        "validation_set": {"mission_count": len(val_records), "runs": val_records},
        "selected_sitl_model": {
            "model_dir": str(model_root),
            "checkpoint_file": "checkpoint.pkl",
            "checkpoint_sha256": train_res["checkpoint_sha256"],
            "manifest": train_res["manifest"],
        },
        "frozen_test_specifications": {
            "mission_count": len(test_specs),
            "missions": test_specs,
            "supported_classes": SITL_CLASSES,
            "unsupported_classes": ["Evil_Twin"],
            "evil_twin_rationale": "MAVLink carries no 802.11 AP/BSSID association semantics; unsupported without external WLAN simulator.",
        },
        "fixed_acceptance_criteria": {
            "fdi_scored_row_precision_min": 0.80,
            "fdi_schedule_wide_end_to_end_recall_min": 0.80,
            "dos_operational_end_to_end_recall_min": 0.70,
            "dos_operational_alert_precision_min": 0.80,
            "replay_precision_min": 0.90,
            "replay_end_to_end_recall_min": 0.90,
            "benign_scored_row_false_positive_rate_max": 0.05,
            "overall_prediction_coverage_min": 0.95,
            "macro_operational_end_to_end_recall_min": 0.85,
            "attack_mission_timely_alert_rate_min": 0.80,
            "false_alert_episodes_per_minute_max": 0.35,
            "mean_attack_alert_delay_max_s": 2.0,
            "single_pass_evaluation_only": True,
            "no_retraining_or_tuning_on_test": True,
        },
    }

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_payload, indent=2) + "\n")
    lock_sha = sha256_file(lock_path)
    (lock_path.parent / (lock_path.name + ".sha256")).write_text(f"{lock_sha}  {lock_path.name}\n")
    print(f"Protocol successfully locked at {lock_path} (SHA-256: {lock_sha})")

    # Step 6: Generate 80 Fresh Locked Test Missions
    print(f"\n=== [5/6] Generating 80 Fresh Locked Test Missions (Parallel 3 Workers) ===")
    test_dirs = execute_parallel_batch(test_specs, test_root, num_workers=3)
    for td in test_dirs:
        reextract_run_features(td)

    # Step 7: Single-Pass Comparative Evaluation
    print(f"\n=== [6/6] Executing Single-Pass Comparative Evaluation Across 4 Models ===")
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Load 4 models
    synth_detector = TransferDetector(ROOT / "models" / "selected")
    v1_detector = TransferDetector(ROOT / "models" / "sitl_selected")
    v2_detector = TransferDetector(ROOT / "models" / "sitl_v2")
    v3_detector = TransferDetector(model_root)

    print("Evaluating Model 1/4: Synthetic Baseline Transfer Checkpoint...")
    synth_runs = [evaluate_run_strictly_causal(td, synth_detector, is_sitl_model=False) for td in test_dirs]
    synth_agg = aggregate_study_metrics(synth_runs)

    print("Evaluating Model 2/4: SITL Iteration 1 Historical Model...")
    v1_runs = [evaluate_run_strictly_causal(td, v1_detector, is_sitl_model=True) for td in test_dirs]
    v1_agg = aggregate_study_metrics(v1_runs)

    print("Evaluating Model 3/4: SITL Iteration 2 Historical Model...")
    v2_runs = [evaluate_run_strictly_causal(td, v2_detector, is_sitl_model=True) for td in test_dirs]
    v2_agg = aggregate_study_metrics(v2_runs)

    print("Evaluating Model 4/4: SITL Iteration 3 Improved Detector Pipeline...")
    v3_runs = [evaluate_run_strictly_causal(td, v3_detector, is_sitl_model=True) for td in test_dirs]
    v3_agg = aggregate_study_metrics(v3_runs)

    (eval_output_dir / "synthetic_transfer_evaluation.json").write_text(json.dumps(synth_agg, indent=2) + "\n")
    (eval_output_dir / "sitl_v1_evaluation.json").write_text(json.dumps(v1_agg, indent=2) + "\n")
    (eval_output_dir / "sitl_v2_evaluation.json").write_text(json.dumps(v2_agg, indent=2) + "\n")
    (eval_output_dir / "sitl_evaluation.json").write_text(json.dumps(v3_agg, indent=2) + "\n")

    # Comparison summary
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mission_matrix": {
            "training_missions": len(train_dirs),
            "validation_missions": len(val_dirs),
            "test_missions": len(test_dirs),
            "total_study_missions": len(train_dirs) + len(val_dirs) + len(test_dirs),
        },
        "total_scheduled_test_observations": v3_agg["total_scheduled_observations"],
        "coverage_rate": v3_agg["overall_coverage_rate"],
        "synthetic_baseline": {
            "benign_fpr": synth_agg["benign_scored_fpr"],
            "fdi_precision": synth_agg["per_class"]["FDI"]["scored_precision"],
            "fdi_recall": synth_agg["per_class"]["FDI"]["operational_recall"],
            "dos_op_recall": synth_agg["per_class"]["DoS"]["operational_recall"],
            "dos_op_precision": synth_agg["per_class"]["DoS"]["operational_precision"],
            "replay_precision": synth_agg["per_class"]["Replay"]["operational_precision"],
            "replay_recall": synth_agg["per_class"]["Replay"]["operational_recall"],
            "macro_op_recall": synth_agg["macro_operational_recall"],
        },
        "sitl_v1_historical": {
            "benign_fpr": v1_agg["benign_scored_fpr"],
            "fdi_precision": v1_agg["per_class"]["FDI"]["scored_precision"],
            "fdi_recall": v1_agg["per_class"]["FDI"]["operational_recall"],
            "dos_op_recall": v1_agg["per_class"]["DoS"]["operational_recall"],
            "dos_op_precision": v1_agg["per_class"]["DoS"]["operational_precision"],
            "replay_precision": v1_agg["per_class"]["Replay"]["operational_precision"],
            "replay_recall": v1_agg["per_class"]["Replay"]["operational_recall"],
            "macro_op_recall": v1_agg["macro_operational_recall"],
        },
        "sitl_v2_historical": {
            "benign_fpr": v2_agg["benign_scored_fpr"],
            "fdi_precision": v2_agg["per_class"]["FDI"]["scored_precision"],
            "fdi_recall": v2_agg["per_class"]["FDI"]["operational_recall"],
            "dos_op_recall": v2_agg["per_class"]["DoS"]["operational_recall"],
            "dos_op_precision": v2_agg["per_class"]["DoS"]["operational_precision"],
            "replay_precision": v2_agg["per_class"]["Replay"]["operational_precision"],
            "replay_recall": v2_agg["per_class"]["Replay"]["operational_recall"],
            "macro_op_recall": v2_agg["macro_operational_recall"],
        },
        "sitl_detector": {
            "benign_fpr": v3_agg["benign_scored_fpr"],
            "fdi_precision": v3_agg["per_class"]["FDI"]["scored_precision"],
            "fdi_recall": v3_agg["per_class"]["FDI"]["operational_recall"],
            "dos_op_recall": v3_agg["per_class"]["DoS"]["operational_recall"],
            "dos_op_precision": v3_agg["per_class"]["DoS"]["operational_precision"],
            "replay_precision": v3_agg["per_class"]["Replay"]["operational_precision"],
            "replay_recall": v3_agg["per_class"]["Replay"]["operational_recall"],
            "macro_op_recall": v3_agg["macro_operational_recall"],
            "latency_s": v3_agg["average_detection_latency_s"],
            "false_alert_episodes_per_minute": v3_agg["benign_false_alert_episodes_per_minute"],
        },
    }
    (eval_output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Export per-mission metrics CSV
    per_mission_rows = []
    for r in v3_runs:
        per_mission_rows.append({
            "run_id": r["run_id"],
            "scheduled": r["scheduled_observations"],
            "covered": r["covered_observations"],
            "coverage_rate": r["coverage_rate"],
            "watchdog_timeout_alerts": r["watchdog_timeout_alerts"],
            "benign_false_alert_episodes": r["benign_false_alert_episodes"],
            "latency_s": r["detection_latency_s"],
            "produced_timely_alert": r["produced_timely_alert"],
        })
    pd.DataFrame(per_mission_rows).to_csv(eval_output_dir / "per_mission_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("=== STUDY EXECUTION COMPLETE ===")
    print(f"Total missions evaluated: {len(test_dirs)} fresh locked missions")
    print("SITL v3 Results:")
    print(f"  FDI Precision:                  {v3_agg['per_class']['FDI']['scored_precision']*100:.2f}% (Target: >=80%)")
    print(f"  FDI Operational Recall:         {v3_agg['per_class']['FDI']['operational_recall']*100:.2f}% (Target: >=80%)")
    print(f"  DoS Operational Recall:         {v3_agg['per_class']['DoS']['operational_recall']*100:.2f}% (Target: >=70%)")
    print(f"  DoS Operational Precision:      {v3_agg['per_class']['DoS']['operational_precision']*100:.2f}% (Target: >=80%)")
    print(f"  Replay Precision:               {v3_agg['per_class']['Replay']['operational_precision']*100:.2f}% (Target: >=90%)")
    print(f"  Replay Operational Recall:      {v3_agg['per_class']['Replay']['operational_recall']*100:.2f}% (Target: >=90%)")
    print(f"  Benign Scored FPR:              {v3_agg['benign_scored_fpr']*100:.2f}% (Target: <=5%)")
    print(f"  Overall Coverage:               {v3_agg['overall_coverage_rate']*100:.2f}% (Target: >=95%)")
    print(f"  Macro Operational Recall:       {v3_agg['macro_operational_recall']*100:.2f}% (Target: >=85%)")
    print(f"  False Alert Episodes / min:     {v3_agg['benign_false_alert_episodes_per_minute']:.2f} (Target: <=0.35)")
    print(f"  Mean Alert Delay:               {v3_agg['average_detection_latency_s']:.2f}s (Target: <=2.0s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
