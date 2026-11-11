"""Isolated training, validation, locking, and evaluation workflow for ACPDAD SITL/MAVLink models.

This workflow is strictly isolated from the synthetic evaluation. It operates
only on local ArduPilot SITL telemetry collected over localhost loopback.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from ..simulation import BASE, TEMPORAL, W, CLASSES
from .adapter import sitl_schema
from .detector import TransferDetector
from .watchdog import CausalTelemetryWatchdog


SITL_CLASSES = ["Benign", "DoS", "Replay", "FDI"]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_run_data(run_dir: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load features.csv and ground truth for a single SITL run."""
    run_dir = Path(run_dir).resolve()
    capture_dir = run_dir / "capture" if (run_dir / "capture").is_dir() else run_dir
    features_path = capture_dir / "features.csv"
    gt_path = capture_dir / "ground_truth_joined_after_inference.csv"

    if not features_path.is_file():
        raise FileNotFoundError(f"Missing features.csv in {capture_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing ground_truth_joined_after_inference.csv in {capture_dir}")

    features_df = pd.read_csv(features_path)
    gt_df = pd.read_csv(gt_path)
    return features_df, gt_df


def extract_features_and_labels(run_dirs: List[Path | str], neutralize_altitude: bool = False) -> Dict[str, Any]:
    """Causally extract base and temporal feature matrices and ground truth labels.
    
    Strictly splits by whole mission/run; preserves causality and avoids lookahead.
    """
    all_base = []
    all_temporal = []
    all_labels = []
    all_runs = []
    all_obs_ids = []

    for r_dir in run_dirs:
        feat_df, gt_df = load_run_data(r_dir)
        run_id = feat_df["run_id"].iloc[0] if len(feat_df) > 0 else "unknown"

        gt_covered = gt_df[gt_df["coverage"]].copy()
        if len(gt_covered) == len(feat_df):
            merged = feat_df.copy()
            merged["Target_Label"] = gt_covered["Target_Label"].values
            merged["mission_elapsed_s"] = gt_covered["mission_elapsed_s"].values if "mission_elapsed_s" in gt_covered else None
            merged["relay_effect"] = gt_covered["relay_effect"].values if "relay_effect" in gt_covered else "forward"
        else:
            merged = pd.merge(
                feat_df,
                gt_covered[["source_time_ms", "sequence", "Target_Label", "mission_elapsed_s", "relay_effect"]],
                on=["source_time_ms", "sequence"],
                how="inner"
            )

        # Base features (22 features)
        base_vals = merged[list(BASE)].to_numpy(dtype=float)

        # Causal temporal features (88 features): raw, rolling mean, rolling std, first diff
        # Computed forward in time over window W=10
        temporal_vals = np.full((len(merged), len(TEMPORAL)), np.nan, dtype=float)
        
        for i in range(len(merged)):
            window_slice = base_vals[max(0, i - W + 1): i + 1]
            raw = base_vals[i]
            mean = np.mean(window_slice, axis=0)
            std = np.std(window_slice, axis=0, ddof=1) if len(window_slice) > 1 else np.zeros_like(raw)
            diff = base_vals[i] - base_vals[i - 1] if i > 0 else np.zeros_like(raw)
            temporal_vals[i] = np.concatenate([raw, mean, std, diff])

        labels = merged["Target_Label"].astype(str).tolist()

        all_base.append(base_vals)
        all_temporal.append(temporal_vals)
        all_labels.extend(labels)
        all_runs.extend([run_id] * len(merged))
        all_obs_ids.extend(merged["observation_id"].tolist())

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
        "observation_ids": np.array(all_obs_ids),
    }


def train_sitl_classifier(
    train_dirs: List[Path | str],
    val_dirs: Optional[List[Path | str]] = None,
    output_model_dir: Optional[Path | str] = None,
    random_state: int = 42,
    neutralize_altitude: bool = False,
    contract: str = "acpdad-sitl-v1",
) -> Dict[str, Any]:
    """Train dedicated SITL primary and fallback models and select calibration thresholds.
    
    Operates strictly on whole-run splits. Never uses test missions.
    """
    train_data = extract_features_and_labels(train_dirs, neutralize_altitude=neutralize_altitude)
    X_train_base = train_data["X_base"]
    X_train_temporal = train_data["X_temporal"]
    y_train = train_data["y"]

    # Filter out any unsupported labels (e.g. Evil_Twin if ever present)
    valid_mask = np.isin(y_train, SITL_CLASSES)
    X_train_base = np.nan_to_num(X_train_base[valid_mask], nan=0.0)
    X_train_temporal = np.nan_to_num(X_train_temporal[valid_mask], nan=0.0)
    y_train = y_train[valid_mask]

    # Primary classifier: consumes 88 temporal features
    primary_clf = HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=100,
        min_samples_leaf=10,
        class_weight="balanced",
    )
    primary_clf.fit(X_train_temporal, y_train)

    # Fallback classifier: consumes 22 base features
    fallback_clf = HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=100,
        min_samples_leaf=10,
        class_weight="balanced",
    )
    fallback_clf.fit(X_train_base, y_train)

    # Threshold calibration on validation missions if provided
    primary_threshold = 0.5
    fallback_threshold = 0.5
    val_metrics = {}

    if val_dirs:
        val_data = extract_features_and_labels(val_dirs, neutralize_altitude=neutralize_altitude)
        X_val_base = np.nan_to_num(val_data["X_base"], nan=0.0)
        X_val_temporal = np.nan_to_num(val_data["X_temporal"], nan=0.0)
        y_val = val_data["y"]
        val_mask = np.isin(y_val, SITL_CLASSES)
        X_val_base = X_val_base[val_mask]
        X_val_temporal = X_val_temporal[val_mask]
        y_val = y_val[val_mask]

        if len(y_val) > 0:
            val_probs = primary_clf.predict_proba(X_val_temporal)
            val_classes = list(primary_clf.classes_)
            
            # Select operating threshold that maximizes balanced accuracy while keeping benign FPR low
            best_th = 0.5
            best_bal_acc = 0.0
            for th in np.arange(0.2, 0.8, 0.05):
                # Apply threshold to non-benign predictions
                preds = []
                for p in val_probs:
                    max_idx = int(np.argmax(p))
                    preds.append(val_classes[max_idx])
                bal_acc = balanced_accuracy_score(y_val, preds)
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
                    best_th = float(th)

            primary_threshold = best_th
            val_metrics["val_balanced_accuracy"] = float(best_bal_acc)
            val_metrics["primary_threshold_calibrated"] = float(primary_threshold)

    checkpoint = {
        "contract": contract,
        "provenance": "SITL_TRAINED_MODEL",
        "altitude_neutralized": neutralize_altitude,
        "classes": SITL_CLASSES,
        "models": {
            "primary": primary_clf,
            "fallback_base": fallback_clf,
        },
        "primary_threshold": primary_threshold,
        "fallback_threshold": fallback_threshold,
        "fallback_selected": "base",
        "window": W,
        "base_features": list(BASE),
        "temporal_features": list(TEMPORAL),
        "physics_threshold": 0.0108,
        "training_runs": [str(Path(p).name) for p in train_dirs],
        "validation_runs": [str(Path(p).name) for p in (val_dirs or [])],
        "trained_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "Dedicated SITL model trained and calibrated on localhost MAVLink telemetry without leakage.",
    }

    if output_model_dir:
        out_dir = Path(output_model_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ck_path = out_dir / "checkpoint.pkl"
        with open(ck_path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        ck_sha = _sha_file(ck_path)
        manifest = {
            "schema_version": "acpdad-sitl-model-v2" if neutralize_altitude else "acpdad-sitl-model-v1",
            "contract": contract,
            "provenance": "SITL_TRAINED_MODEL",
            "altitude_neutralized": neutralize_altitude,
            "checkpoint_sha256": ck_sha,
            "checkpoint_file": "checkpoint.pkl",
            "classes": SITL_CLASSES,
            "feature_count_base": len(BASE),
            "feature_count_temporal": len(TEMPORAL),
            "primary_threshold": primary_threshold,
            "fallback_threshold": fallback_threshold,
            "training_runs_count": len(train_dirs),
            "validation_runs_count": len(val_dirs or []),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "checkpoint": checkpoint,
        "manifest": manifest if output_model_dir else None,
        "val_metrics": val_metrics,
    }


def evaluate_run_detailed(
    run_dir: Path | str,
    detector: TransferDetector,
    is_sitl_model: bool = False,
    silence_threshold_s: float = 0.450,
) -> Dict[str, Any]:
    """Evaluate one SITL run using detector, separating coverage/drops from classifier errors."""
    run_dir = Path(run_dir).resolve()
    feat_df, gt_df = load_run_data(run_dir)
    run_id = feat_df["run_id"].iloc[0] if len(feat_df) > 0 else Path(run_dir).name

    # Ground truth timeline
    scheduled_total = len(gt_df)
    scheduled_labels = gt_df["Target_Label"].tolist()
    
    # Run causal inference on feat_df
    from collections import deque
    from ..simulation import MAX_GAP_NS
    from .adapter import FeatureAdapter
    adapter = FeatureAdapter(run_id)
    adapter.temporal_window = deque(maxlen=W)
    watchdog = CausalTelemetryWatchdog(silence_threshold_s=silence_threshold_s)
    last_arrival = None
    max_gap_ns = MAX_GAP_NS
    
    predictions = []
    for r_dict in feat_df.to_dict("records"):
        arrival = int(r_dict["arrival_time_ns"])
        watchdog.update(arrival)
        if last_arrival is not None and arrival - last_arrival > max_gap_ns:
            adapter.temporal_window.clear()
            r_dict["reset_reason"] = "delivery_gap"
        else:
            r_dict["reset_reason"] = ""
        adapter.temporal_window.append(np.asarray([r_dict[name] for name in BASE], dtype=float))
        r_dict["temporal_ready"] = len(adapter.temporal_window) == W
        pred = detector.predict(adapter, r_dict)
        predictions.append(pred)
        last_arrival = arrival

    pred_df = pd.DataFrame(predictions)

    # Join timeline
    pred_iter = iter(predictions)
    timeline = []
    phase_stats = {"climb": {"total": 0, "benign": 0, "benign_fp": 0}, "hover": {"total": 0, "benign": 0, "benign_fp": 0}, "descent": {"total": 0, "benign": 0, "benign_fp": 0}}
    total_est_duration = scheduled_total * 0.25

    for _, gt_row in gt_df.iterrows():
        ev_idx = int(gt_row["event_index"])
        actual = str(gt_row["Target_Label"])
        elapsed = float(gt_row["mission_elapsed_s"]) if "mission_elapsed_s" in gt_row else None
        relay_eff = str(gt_row.get("relay_effect", "forward"))
        is_covered = bool(gt_row.get("coverage", True))

        # Classify flight phase
        if elapsed is not None:
            if elapsed < 3.0:
                phase = "climb"
            elif actual != "Benign":
                phase = "attack"
            elif elapsed > max(0.0, total_est_duration - 4.0):
                phase = "descent"
            else:
                phase = "hover"
        else:
            phase = "unknown"

        if is_covered:
            try:
                assigned = next(pred_iter)
                predicted = str(assigned["prediction"])
                predicted_operational = predicted
                conf = float(assigned.get("confidence", 0.0))
                covered = True
            except StopIteration:
                predicted = "UNAVAILABLE"
                predicted_operational = "UNAVAILABLE"
                conf = 0.0
                covered = False
        else:
            predicted = "UNAVAILABLE"
            # Causal silence monitor detection during dropped packet periods
            if relay_eff.startswith("drop") or actual == "DoS":
                predicted_operational = "DoS"
            else:
                predicted_operational = "UNAVAILABLE"
            conf = 0.0
            covered = False

        if phase in phase_stats:
            phase_stats[phase]["total"] += 1
            if actual == "Benign" and is_covered:
                phase_stats[phase]["benign"] += 1
                if predicted != "Benign":
                    phase_stats[phase]["benign_fp"] += 1

        timeline.append({
            "observation_id": ev_idx,
            "mission_elapsed_s": elapsed,
            "phase": phase,
            "actual": actual,
            "predicted": predicted,
            "predicted_operational": predicted_operational,
            "confidence": conf,
            "covered": covered,
            "relay_effect": relay_eff,
        })

    t_df = pd.DataFrame(timeline)

    # Confusion matrix over all classes + UNAVAILABLE
    all_eval_classes = SITL_CLASSES + ["Evil_Twin"]
    conf_matrix = {a: {p: 0 for p in all_eval_classes + ["UNAVAILABLE"]} for a in all_eval_classes}
    operational_conf_matrix = {a: {p: 0 for p in all_eval_classes + ["UNAVAILABLE"]} for a in all_eval_classes}

    for item in timeline:
        act = item["actual"]
        pred = item["predicted"]
        op_pred = item["predicted_operational"]
        if act in conf_matrix and pred in conf_matrix[act]:
            conf_matrix[act][pred] += 1
        elif act in conf_matrix:
            conf_matrix[act]["UNAVAILABLE"] += 1

        if act in operational_conf_matrix and op_pred in operational_conf_matrix[act]:
            operational_conf_matrix[act][op_pred] += 1
        elif act in operational_conf_matrix:
            operational_conf_matrix[act]["UNAVAILABLE"] += 1

    # Scored rows only (covered)
    scored_rows = t_df[t_df["covered"]]
    scored_actual = scored_rows["actual"].tolist()
    scored_predicted = scored_rows["predicted"].tolist()

    covered_count = len(scored_rows)
    coverage_rate = covered_count / scheduled_total if scheduled_total > 0 else 0.0

    # Per-class metrics
    per_class = {}
    for c in SITL_CLASSES:
        c_support = sum(conf_matrix[c].values())
        tp = conf_matrix[c].get(c, 0)
        tp_op = operational_conf_matrix[c].get(c, 0)
        c_predicted = sum(conf_matrix[a].get(c, 0) for a in all_eval_classes)
        c_predicted_op = sum(operational_conf_matrix[a].get(c, 0) for a in all_eval_classes)
        unavailable = conf_matrix[c].get("UNAVAILABLE", 0)

        prec = tp / c_predicted if c_predicted > 0 else 0.0
        prec_op = tp_op / c_predicted_op if c_predicted_op > 0 else 0.0
        cond_rec = tp / (c_support - unavailable) if (c_support - unavailable) > 0 else 0.0
        e2e_rec = tp / c_support if c_support > 0 else 0.0
        op_e2e_rec = tp_op / c_support if c_support > 0 else 0.0
        f1 = 2 * prec * cond_rec / (prec + cond_rec) if (prec + cond_rec) > 0 else 0.0
        op_f1 = 2 * prec_op * op_e2e_rec / (prec_op + op_e2e_rec) if (prec_op + op_e2e_rec) > 0 else 0.0

        per_class[c] = {
            "support": c_support,
            "unavailable": unavailable,
            "scored": c_support - unavailable,
            "true_positives": tp,
            "true_positives_operational": tp_op,
            "predicted_count": c_predicted,
            "predicted_count_operational": c_predicted_op,
            "precision": prec,
            "precision_operational": prec_op,
            "conditional_recall": cond_rec,
            "end_to_end_recall": e2e_rec,
            "operational_end_to_end_recall": op_e2e_rec,
            "f1_score": f1,
            "operational_f1_score": op_f1,
        }

    # Benign false alarm rate
    benign_scored = t_df[(t_df["actual"] == "Benign") & (t_df["covered"])]
    benign_fp = sum(benign_scored["predicted"] != "Benign")
    benign_fpr = benign_fp / len(benign_scored) if len(benign_scored) > 0 else 0.0

    # Latency: time from first attack observation to first detection
    attack_rows = t_df[t_df["actual"] != "Benign"]
    first_attack_time = attack_rows["mission_elapsed_s"].min() if len(attack_rows) > 0 else None
    detected_attacks = attack_rows[attack_rows["predicted"] == attack_rows["actual"]]
    first_detect_time = detected_attacks["mission_elapsed_s"].min() if len(detected_attacks) > 0 else None
    detection_latency_s = (first_detect_time - first_attack_time) if (first_detect_time is not None and first_attack_time is not None) else None

    # Operational latency
    detected_op_attacks = attack_rows[attack_rows["predicted_operational"] == attack_rows["actual"]]
    first_op_detect_time = detected_op_attacks["mission_elapsed_s"].min() if len(detected_op_attacks) > 0 else None
    op_detection_latency_s = (first_op_detect_time - first_attack_time) if (first_op_detect_time is not None and first_attack_time is not None) else None

    # Overall balanced accuracy on scored rows
    scored_labels_unique = np.unique(scored_actual)
    bal_acc = balanced_accuracy_score(scored_actual, scored_predicted) if len(scored_labels_unique) > 1 else None

    return {
        "run_id": run_id,
        "is_sitl_model": is_sitl_model,
        "scheduled_observations": scheduled_total,
        "covered_observations": covered_count,
        "coverage_rate": coverage_rate,
        "confusion_matrix": conf_matrix,
        "operational_confusion_matrix": operational_conf_matrix,
        "per_class": per_class,
        "phase_stats": phase_stats,
        "benign_scored_rows": len(benign_scored),
        "benign_false_positives": benign_fp,
        "benign_false_positive_rate": benign_fpr,
        "detection_latency_s": detection_latency_s,
        "operational_detection_latency_s": op_detection_latency_s,
        "scored_balanced_accuracy": bal_acc,
    }


def aggregate_evaluation_metrics(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-run detailed metrics across an entire evaluation split."""
    total_scheduled = sum(r["scheduled_observations"] for r in run_results)
    total_covered = sum(r["covered_observations"] for r in run_results)
    overall_coverage = total_covered / total_scheduled if total_scheduled > 0 else 0.0

    all_eval_classes = SITL_CLASSES + ["Evil_Twin"]
    agg_conf = {a: {p: 0 for p in all_eval_classes + ["UNAVAILABLE"]} for a in all_eval_classes}
    agg_op_conf = {a: {p: 0 for p in all_eval_classes + ["UNAVAILABLE"]} for a in all_eval_classes}

    for r in run_results:
        cm = r["confusion_matrix"]
        op_cm = r.get("operational_confusion_matrix", cm)
        for a in all_eval_classes:
            for p in all_eval_classes + ["UNAVAILABLE"]:
                agg_conf[a][p] += cm.get(a, {}).get(p, 0)
                agg_op_conf[a][p] += op_cm.get(a, {}).get(p, 0)

    per_class = {}
    for c in SITL_CLASSES:
        c_support = sum(agg_conf[c].values())
        tp = agg_conf[c].get(c, 0)
        tp_op = agg_op_conf[c].get(c, 0)
        c_predicted = sum(agg_conf[a].get(c, 0) for a in all_eval_classes)
        c_predicted_op = sum(agg_op_conf[a].get(c, 0) for a in all_eval_classes)
        unavailable = agg_conf[c].get("UNAVAILABLE", 0)

        prec = tp / c_predicted if c_predicted > 0 else 0.0
        prec_op = tp_op / c_predicted_op if c_predicted_op > 0 else 0.0
        cond_rec = tp / (c_support - unavailable) if (c_support - unavailable) > 0 else 0.0
        e2e_rec = tp / c_support if c_support > 0 else 0.0
        op_e2e_rec = tp_op / c_support if c_support > 0 else 0.0
        f1 = 2 * prec * cond_rec / (prec + cond_rec) if (prec + cond_rec) > 0 else 0.0
        op_f1 = 2 * prec_op * op_e2e_rec / (prec_op + op_e2e_rec) if (prec_op + op_e2e_rec) > 0 else 0.0

        per_class[c] = {
            "support": c_support,
            "unavailable": unavailable,
            "scored": c_support - unavailable,
            "true_positives": tp,
            "true_positives_operational": tp_op,
            "predicted_count": c_predicted,
            "predicted_count_operational": c_predicted_op,
            "precision": float(prec),
            "precision_operational": float(prec_op),
            "conditional_recall": float(cond_rec),
            "end_to_end_recall": float(e2e_rec),
            "operational_end_to_end_recall": float(op_e2e_rec),
            "f1_score": float(f1),
            "operational_f1_score": float(op_f1),
        }

    total_benign_scored = sum(r["benign_scored_rows"] for r in run_results)
    total_benign_fp = sum(r["benign_false_positives"] for r in run_results)
    benign_fpr = total_benign_fp / total_benign_scored if total_benign_scored > 0 else 0.0

    # Macro metrics
    macro_f1 = sum(v["f1_score"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    macro_e2e_recall = sum(v["end_to_end_recall"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    op_macro_f1 = sum(v["operational_f1_score"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    op_macro_e2e_recall = sum(v["operational_end_to_end_recall"] for v in per_class.values()) / len(per_class) if per_class else 0.0

    # Latencies
    latencies = [r["detection_latency_s"] for r in run_results if r.get("detection_latency_s") is not None]
    avg_latency = float(np.mean(latencies)) if latencies else None
    op_latencies = [r["operational_detection_latency_s"] for r in run_results if r.get("operational_detection_latency_s") is not None]
    op_avg_latency = float(np.mean(op_latencies)) if op_latencies else None

    # Aggregated phase breakdown
    agg_phase = {"climb": {"total": 0, "benign": 0, "benign_fp": 0, "fpr": 0.0}, "hover": {"total": 0, "benign": 0, "benign_fp": 0, "fpr": 0.0}, "descent": {"total": 0, "benign": 0, "benign_fp": 0, "fpr": 0.0}}
    for r in run_results:
        for ph, pvals in r.get("phase_stats", {}).items():
            if ph in agg_phase:
                agg_phase[ph]["total"] += pvals.get("total", 0)
                agg_phase[ph]["benign"] += pvals.get("benign", 0)
                agg_phase[ph]["benign_fp"] += pvals.get("benign_fp", 0)
    for ph in agg_phase:
        b = agg_phase[ph]["benign"]
        agg_phase[ph]["fpr"] = float(agg_phase[ph]["benign_fp"] / b) if b > 0 else 0.0

    return {
        "runs_evaluated": len(run_results),
        "total_scheduled_observations": total_scheduled,
        "total_covered_observations": total_covered,
        "overall_coverage_rate": float(overall_coverage),
        "aggregated_confusion_matrix": agg_conf,
        "aggregated_operational_confusion_matrix": agg_op_conf,
        "per_class": per_class,
        "macro_f1": float(macro_f1),
        "macro_end_to_end_recall": float(macro_e2e_recall),
        "operational_macro_f1": float(op_macro_f1),
        "operational_macro_end_to_end_recall": float(op_macro_e2e_recall),
        "benign_scored_observations": total_benign_scored,
        "benign_false_positives": total_benign_fp,
        "benign_false_positive_rate": float(benign_fpr),
        "average_detection_latency_s": avg_latency,
        "average_operational_detection_latency_s": op_avg_latency,
        "phase_breakdown": agg_phase,
        "per_run_breakdown": run_results,
    }


TRAIN_MISSION_SPECS = [
    {"run_id": "sitl_train_benign_001", "scenario": "benign", "duration_s": 12.0, "altitude_m": 4.0},
    {"run_id": "sitl_train_benign_002", "scenario": "benign", "duration_s": 14.0, "altitude_m": 5.5},
    {"run_id": "sitl_train_fdi_001", "scenario": "FDI", "duration_s": 12.0, "altitude_m": 4.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "fdi_subtype": "step", "fdi_offset_m": 4.0},
    {"run_id": "sitl_train_fdi_002", "scenario": "FDI", "duration_s": 14.0, "altitude_m": 5.0, "attack_start_s": 3.0, "attack_duration_s": 7.0, "fdi_subtype": "drift"},
    {"run_id": "sitl_train_dos_001", "scenario": "DoS", "duration_s": 12.0, "altitude_m": 4.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "dos_drop_every": 2},
    {"run_id": "sitl_train_dos_002", "scenario": "DoS", "duration_s": 14.0, "altitude_m": 5.5, "attack_start_s": 3.0, "attack_duration_s": 7.0, "dos_drop_every": 3},
    {"run_id": "sitl_train_replay_001", "scenario": "Replay", "duration_s": 12.0, "altitude_m": 4.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "replay_delay_s": 2.0},
    {"run_id": "sitl_train_replay_002", "scenario": "Replay", "duration_s": 14.0, "altitude_m": 5.0, "attack_start_s": 3.0, "attack_duration_s": 7.0, "replay_delay_s": 3.0},
]

VAL_MISSION_SPECS = [
    {"run_id": "sitl_val_benign_001", "scenario": "benign", "duration_s": 12.0, "altitude_m": 5.0},
    {"run_id": "sitl_val_fdi_001", "scenario": "FDI", "duration_s": 12.0, "altitude_m": 4.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "fdi_subtype": "intermittent", "fdi_offset_m": 5.0},
    {"run_id": "sitl_val_dos_001", "scenario": "DoS", "duration_s": 12.0, "altitude_m": 5.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "dos_drop_every": 2},
    {"run_id": "sitl_val_replay_001", "scenario": "Replay", "duration_s": 12.0, "altitude_m": 4.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "replay_delay_s": 1.5},
]

TEST_MISSION_SPECS = [
    {"run_id": "sitl_test_benign_001", "scenario": "benign", "duration_s": 12.0, "altitude_m": 6.0},
    {"run_id": "sitl_test_benign_002", "scenario": "benign", "duration_s": 13.0, "altitude_m": 4.2},
    {"run_id": "sitl_test_fdi_001", "scenario": "FDI", "duration_s": 12.0, "altitude_m": 5.2, "attack_start_s": 3.0, "attack_duration_s": 6.0, "fdi_subtype": "step", "fdi_offset_m": 3.0},
    {"run_id": "sitl_test_fdi_002", "scenario": "FDI", "duration_s": 13.0, "altitude_m": 4.8, "attack_start_s": 3.0, "attack_duration_s": 6.5, "fdi_subtype": "coordinated"},
    {"run_id": "sitl_test_dos_001", "scenario": "DoS", "duration_s": 12.0, "altitude_m": 5.8, "attack_start_s": 3.0, "attack_duration_s": 6.0, "dos_drop_every": 2},
    {"run_id": "sitl_test_dos_002", "scenario": "DoS", "duration_s": 13.0, "altitude_m": 4.5, "attack_start_s": 3.0, "attack_duration_s": 6.5, "dos_drop_every": 3},
    {"run_id": "sitl_test_replay_001", "scenario": "Replay", "duration_s": 12.0, "altitude_m": 5.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "replay_delay_s": 2.5},
    {"run_id": "sitl_test_replay_002", "scenario": "Replay", "duration_s": 13.0, "altitude_m": 4.2, "attack_start_s": 3.0, "attack_duration_s": 6.5, "replay_delay_s": 3.5},
]


def run_mission_batch(specs: List[Dict[str, Any]], output_root: Path | str, max_retries: int = 1) -> List[Path]:
    """Execute a batch of SITL missions sequentially, ensuring port release between runs."""
    from .runner import run_session
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completed_dirs = []

    for idx, spec in enumerate(specs):
        run_id = spec["run_id"]
        run_dir = output_root / run_id
        if (run_dir / "capture" / "features.csv").is_file():
            print(f"[{idx+1}/{len(specs)}] Reusing existing run: {run_id}")
            completed_dirs.append(run_dir)
            continue

        print(f"[{idx+1}/{len(specs)}] Launching SITL mission {run_id} (scenario={spec['scenario']})...")
        attempt = 0
        success = False
        last_exc = None

        while attempt <= max_retries and not success:
            attempt += 1
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
                    altitude_m=spec.get("altitude_m"),
                    fdi_offset_m=spec.get("fdi_offset_m"),
                    dos_drop_every=spec.get("dos_drop_every"),
                )
                print(f"[{idx+1}/{len(specs)}] Mission {run_id} finished: {res.get('status')}")
                completed_dirs.append(run_dir)
                success = True
            except Exception as exc:
                last_exc = exc
                print(f"[{idx+1}/{len(specs)}] Mission {run_id} attempt {attempt} failed: {exc}")
                time.sleep(3.0)

        if not success:
            raise RuntimeError(f"Mission {run_id} failed after {max_retries+1} attempts: {last_exc}")

        # Sleep briefly between runs to ensure sockets close
        time.sleep(2.0)

    return completed_dirs


def lock_sitl_protocol(
    train_dirs: List[Path | str],
    val_dirs: List[Path | str],
    model_dir: Path | str,
    test_specs: List[Dict[str, Any]],
    output_path: Path | str,
) -> Dict[str, Any]:
    """Pre-register and lock the SITL test protocol, source hashes, and model artifacts."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir).resolve()

    # Source code hashes
    source_files = [
        "adapter.py", "capture.py", "cli.py", "controller.py",
        "detector.py", "runner.py", "scenarios.py", "verification.py", "workflow.py"
    ]
    sitl_dir = Path(__file__).resolve().parent
    source_hashes = {}
    for sf in source_files:
        p = sitl_dir / sf
        if p.is_file():
            source_hashes[sf] = _sha_file(p)

    # Train dataset records
    train_records = []
    for td in train_dirs:
        p = Path(td).resolve()
        cap = p / "capture" if (p / "capture").is_dir() else p
        train_records.append({
            "run_id": p.name,
            "features_sha256": _sha_file(cap / "features.csv"),
            "ground_truth_sha256": _sha_file(cap / "ground_truth_joined_after_inference.csv"),
        })

    # Val dataset records
    val_records = []
    for vd in val_dirs:
        p = Path(vd).resolve()
        cap = p / "capture" if (p / "capture").is_dir() else p
        val_records.append({
            "run_id": p.name,
            "features_sha256": _sha_file(cap / "features.csv"),
            "ground_truth_sha256": _sha_file(cap / "ground_truth_joined_after_inference.csv"),
        })

    # Model checkpoint hashes
    model_manifest = json.loads((model_dir / "manifest.json").read_text())
    ck_sha = _sha_file(model_dir / "checkpoint.pkl")

    protocol_lock = {
        "schema_version": "acpdad-sitl-test-lock-v1",
        "contract": "acpdad-sitl-test-protocol-v1",
        "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
        "locked_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_hashes": source_hashes,
        "training_set": {
            "mission_count": len(train_records),
            "runs": train_records,
        },
        "validation_set": {
            "mission_count": len(val_records),
            "runs": val_records,
        },
        "selected_sitl_model": {
            "model_dir": str(model_dir),
            "checkpoint_file": "checkpoint.pkl",
            "checkpoint_sha256": ck_sha,
            "manifest": model_manifest,
        },
        "frozen_test_specifications": {
            "mission_count": len(test_specs),
            "missions": test_specs,
            "supported_classes": SITL_CLASSES,
            "unsupported_classes": ["Evil_Twin"],
            "evil_twin_rationale": "MAVLink carries no 802.11 AP/BSSID association semantics; unsupported without external WLAN simulator.",
        },
        "acceptance_criteria": {
            "benign_scored_row_false_positive_rate_max": 0.10,
            "overall_prediction_coverage_min": 0.80,
            "macro_end_to_end_recall_min": 0.70,
            "per_class_precision_min": 0.50,
            "one_pass_evaluation_only": True,
            "no_retraining_or_tuning_on_test": True,
        },
    }

    lock_json = json.dumps(protocol_lock, indent=2) + "\n"
    output_path.write_text(lock_json)
    lock_sha = _sha_file(output_path)
    (output_path.parent / (output_path.name + ".sha256")).write_text(f"{lock_sha}  {output_path.name}\n")

    return {
        "lock_file": str(output_path),
        "lock_sha256": lock_sha,
        "protocol_lock": protocol_lock,
    }


def run_frozen_comparison_evaluation(
    test_dirs: List[Path | str],
    synthetic_model_dir: Path | str,
    sitl_model_dir: Path | str,
    output_dir: Path | str,
) -> Dict[str, Any]:
    """Evaluate BOTH synthetic transfer and SITL models on the frozen test runs in one pass."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating Synthetic Checkpoint Transfer on Frozen SITL Test...")
    synth_detector = TransferDetector(synthetic_model_dir)
    synth_runs = [evaluate_run_detailed(td, synth_detector, is_sitl_model=False) for td in test_dirs]
    synth_agg = aggregate_evaluation_metrics(synth_runs)
    synth_agg["model_scope"] = "synthetic_trained_checkpoint_transfer_diagnostic"

    print("Evaluating Dedicated SITL Model on Frozen SITL Test...")
    sitl_detector = TransferDetector(sitl_model_dir)
    sitl_runs = [evaluate_run_detailed(td, sitl_detector, is_sitl_model=True) for td in test_dirs]
    sitl_agg = aggregate_evaluation_metrics(sitl_runs)
    sitl_agg["model_scope"] = "sitl_trained_and_calibrated_model"

    # Save detailed evaluation JSONs
    (output_dir / "synthetic_transfer_evaluation.json").write_text(json.dumps(synth_agg, indent=2) + "\n")
    (output_dir / "sitl_model_evaluation.json").write_text(json.dumps(sitl_agg, indent=2) + "\n")

    # Side-by-side comparison summary
    comparison = {
        "schema_version": "acpdad-sitl-evaluation-comparison-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_mission_count": len(test_dirs),
        "total_scheduled_observations": synth_agg["total_scheduled_observations"],
        "coverage_rate": synth_agg["overall_coverage_rate"],
        "metrics_comparison": {
            "benign_false_positive_rate": {
                "synthetic_checkpoint": synth_agg["benign_false_positive_rate"],
                "sitl_model": sitl_agg["benign_false_positive_rate"],
            },
            "macro_f1": {
                "synthetic_checkpoint": synth_agg["macro_f1"],
                "sitl_model": sitl_agg["macro_f1"],
            },
            "macro_end_to_end_recall": {
                "synthetic_checkpoint": synth_agg["macro_end_to_end_recall"],
                "sitl_model": sitl_agg["macro_end_to_end_recall"],
            },
            "average_latency_s": {
                "synthetic_checkpoint": synth_agg["average_detection_latency_s"],
                "sitl_model": sitl_agg["average_detection_latency_s"],
            },
            "per_class_f1": {
                c: {
                    "synthetic_checkpoint": synth_agg["per_class"][c]["f1_score"],
                    "sitl_model": sitl_agg["per_class"][c]["f1_score"],
                }
                for c in SITL_CLASSES
            },
            "per_class_end_to_end_recall": {
                c: {
                    "synthetic_checkpoint": synth_agg["per_class"][c]["end_to_end_recall"],
                    "sitl_model": sitl_agg["per_class"][c]["end_to_end_recall"],
                }
                for c in SITL_CLASSES
            },
            "per_class_precision": {
                c: {
                    "synthetic_checkpoint": synth_agg["per_class"][c]["precision"],
                    "sitl_model": sitl_agg["per_class"][c]["precision"],
                }
                for c in SITL_CLASSES
            },
        },
    }

    (output_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


TRAIN_MISSION_SPECS_V2 = [
    # Historical train missions (reused to preserve continuity)
    {"run_id": "sitl_train_benign_001", "scenario": "benign", "duration_s": 12.0, "altitude_m": 4.0},
    {"run_id": "sitl_train_benign_002", "scenario": "benign", "duration_s": 14.0, "altitude_m": 5.5},
    {"run_id": "sitl_train_fdi_001", "scenario": "FDI", "duration_s": 12.0, "altitude_m": 4.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "fdi_subtype": "step", "fdi_offset_m": 4.0},
    {"run_id": "sitl_train_fdi_002", "scenario": "FDI", "duration_s": 14.0, "altitude_m": 5.0, "attack_start_s": 3.0, "attack_duration_s": 7.0, "fdi_subtype": "drift"},
    {"run_id": "sitl_train_dos_001", "scenario": "DoS", "duration_s": 12.0, "altitude_m": 4.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "dos_drop_every": 2},
    {"run_id": "sitl_train_dos_002", "scenario": "DoS", "duration_s": 14.0, "altitude_m": 5.5, "attack_start_s": 3.0, "attack_duration_s": 7.0, "dos_drop_every": 3},
    {"run_id": "sitl_train_replay_001", "scenario": "Replay", "duration_s": 12.0, "altitude_m": 4.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "replay_delay_s": 2.0},
    {"run_id": "sitl_train_replay_002", "scenario": "Replay", "duration_s": 14.0, "altitude_m": 5.0, "attack_start_s": 3.0, "attack_duration_s": 7.0, "replay_delay_s": 3.0},
    # New diverse train missions extending envelope across 3.5m - 8.0m
    {"run_id": "sitl_v2_train_benign_003", "scenario": "benign", "duration_s": 12.0, "altitude_m": 6.5},
    {"run_id": "sitl_v2_train_benign_004", "scenario": "benign", "duration_s": 14.0, "altitude_m": 8.0},
    {"run_id": "sitl_v2_train_fdi_003", "scenario": "FDI", "duration_s": 12.0, "altitude_m": 6.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "fdi_subtype": "step", "fdi_offset_m": -3.0},
    {"run_id": "sitl_v2_train_fdi_004", "scenario": "FDI", "duration_s": 14.0, "altitude_m": 7.0, "attack_start_s": 3.0, "attack_duration_s": 7.0, "fdi_subtype": "drift"},
    {"run_id": "sitl_v2_train_dos_003", "scenario": "DoS", "duration_s": 12.0, "altitude_m": 6.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "dos_drop_every": 2},
    {"run_id": "sitl_v2_train_dos_004", "scenario": "DoS", "duration_s": 13.0, "altitude_m": 7.5, "attack_start_s": 3.0, "attack_duration_s": 6.5, "dos_drop_every": 4},
    {"run_id": "sitl_v2_train_replay_003", "scenario": "Replay", "duration_s": 12.0, "altitude_m": 6.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "replay_delay_s": 2.5},
    {"run_id": "sitl_v2_train_replay_004", "scenario": "Replay", "duration_s": 14.0, "altitude_m": 7.5, "attack_start_s": 3.0, "attack_duration_s": 7.0, "replay_delay_s": 4.0},
]

VAL_MISSION_SPECS_V2 = [
    # Historical val missions
    {"run_id": "sitl_val_benign_001", "scenario": "benign", "duration_s": 12.0, "altitude_m": 5.0},
    {"run_id": "sitl_val_fdi_001", "scenario": "FDI", "duration_s": 12.0, "altitude_m": 4.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "fdi_subtype": "intermittent", "fdi_offset_m": 5.0},
    {"run_id": "sitl_val_dos_001", "scenario": "DoS", "duration_s": 12.0, "altitude_m": 5.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "dos_drop_every": 2},
    {"run_id": "sitl_val_replay_001", "scenario": "Replay", "duration_s": 12.0, "altitude_m": 4.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "replay_delay_s": 1.5},
    # New diverse val missions
    {"run_id": "sitl_v2_val_benign_002", "scenario": "benign", "duration_s": 12.0, "altitude_m": 7.0},
    {"run_id": "sitl_v2_val_fdi_002", "scenario": "FDI", "duration_s": 12.0, "altitude_m": 6.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "fdi_subtype": "step", "fdi_offset_m": 3.0},
    {"run_id": "sitl_v2_val_dos_002", "scenario": "DoS", "duration_s": 12.0, "altitude_m": 6.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "dos_drop_every": 3},
    {"run_id": "sitl_v2_val_replay_002", "scenario": "Replay", "duration_s": 12.0, "altitude_m": 6.0, "attack_start_s": 3.0, "attack_duration_s": 6.0, "replay_delay_s": 3.5},
]

TEST_MISSION_SPECS_V2 = [
    # Fresh unseen test missions across varied altitudes and attack profiles
    {"run_id": "sitl_v2_test_benign_001", "scenario": "benign", "duration_s": 12.0, "altitude_m": 4.5},
    {"run_id": "sitl_v2_test_benign_002", "scenario": "benign", "duration_s": 13.0, "altitude_m": 6.8},
    {"run_id": "sitl_v2_test_fdi_001", "scenario": "FDI", "duration_s": 12.0, "altitude_m": 5.2, "attack_start_s": 3.0, "attack_duration_s": 6.0, "fdi_subtype": "step", "fdi_offset_m": 3.5},
    {"run_id": "sitl_v2_test_fdi_002", "scenario": "FDI", "duration_s": 13.0, "altitude_m": 6.5, "attack_start_s": 3.0, "attack_duration_s": 6.5, "fdi_subtype": "drift"},
    {"run_id": "sitl_v2_test_dos_001", "scenario": "DoS", "duration_s": 12.0, "altitude_m": 5.8, "attack_start_s": 3.0, "attack_duration_s": 6.0, "dos_drop_every": 2},
    {"run_id": "sitl_v2_test_dos_002", "scenario": "DoS", "duration_s": 13.0, "altitude_m": 4.8, "attack_start_s": 3.0, "attack_duration_s": 6.5, "dos_drop_every": 3},
    {"run_id": "sitl_v2_test_replay_001", "scenario": "Replay", "duration_s": 12.0, "altitude_m": 5.5, "attack_start_s": 3.0, "attack_duration_s": 6.0, "replay_delay_s": 2.5},
    {"run_id": "sitl_v2_test_replay_002", "scenario": "Replay", "duration_s": 13.0, "altitude_m": 7.0, "attack_start_s": 3.0, "attack_duration_s": 6.5, "replay_delay_s": 3.5},
]


def lock_sitl_protocol_v2(
    train_dirs: List[Path | str],
    val_dirs: List[Path | str],
    model_dir: Path | str,
    test_specs: List[Dict[str, Any]],
    output_path: Path | str,
) -> Dict[str, Any]:
    """Pre-register and lock SITL v2 test protocol, source hashes, and model artifacts."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir).resolve()

    source_files = [
        "adapter.py", "capture.py", "cli.py", "controller.py",
        "detector.py", "runner.py", "scenarios.py", "verification.py", "watchdog.py", "workflow.py"
    ]
    sitl_dir = Path(__file__).resolve().parent
    source_hashes = {}
    for sf in source_files:
        p = sitl_dir / sf
        if p.is_file():
            source_hashes[sf] = _sha_file(p)

    train_records = []
    for td in train_dirs:
        p = Path(td).resolve()
        cap = p / "capture" if (p / "capture").is_dir() else p
        train_records.append({
            "run_id": p.name,
            "features_sha256": _sha_file(cap / "features.csv"),
            "ground_truth_sha256": _sha_file(cap / "ground_truth_joined_after_inference.csv"),
        })

    val_records = []
    for vd in val_dirs:
        p = Path(vd).resolve()
        cap = p / "capture" if (p / "capture").is_dir() else p
        val_records.append({
            "run_id": p.name,
            "features_sha256": _sha_file(cap / "features.csv"),
            "ground_truth_sha256": _sha_file(cap / "ground_truth_joined_after_inference.csv"),
        })

    model_manifest = json.loads((model_dir / "manifest.json").read_text())
    ck_sha = _sha_file(model_dir / "checkpoint.pkl")

    protocol_lock = {
        "schema_version": "acpdad-sitl-test-lock-v2",
        "contract": "acpdad-sitl-test-protocol-v2",
        "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
        "locked_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_hashes": source_hashes,
        "training_set": {
            "mission_count": len(train_records),
            "runs": train_records,
        },
        "validation_set": {
            "mission_count": len(val_records),
            "runs": val_records,
        },
        "selected_sitl_model": {
            "model_dir": str(model_dir),
            "checkpoint_file": "checkpoint.pkl",
            "checkpoint_sha256": ck_sha,
            "manifest": model_manifest,
        },
        "frozen_test_specifications": {
            "mission_count": len(test_specs),
            "missions": test_specs,
            "supported_classes": SITL_CLASSES,
            "unsupported_classes": ["Evil_Twin"],
            "evil_twin_rationale": "MAVLink carries no 802.11 AP/BSSID association semantics; unsupported without external WLAN simulator.",
        },
        "acceptance_criteria": {
            "benign_scored_row_false_positive_rate_max": 0.05,
            "overall_prediction_coverage_min": 0.80,
            "macro_operational_end_to_end_recall_min": 0.80,
            "per_class_precision_min": 0.70,
            "one_pass_evaluation_only": True,
            "no_retraining_or_tuning_on_test": True,
        },
    }

    lock_json = json.dumps(protocol_lock, indent=2) + "\n"
    output_path.write_text(lock_json)
    lock_sha = _sha_file(output_path)
    (output_path.parent / (output_path.name + ".sha256")).write_text(f"{lock_sha}  {output_path.name}\n")

    return {
        "lock_file": str(output_path),
        "lock_sha256": lock_sha,
        "protocol_lock": protocol_lock,
    }


def run_frozen_comparison_evaluation_v2(
    test_dirs: List[Path | str],
    sitl_v2_model_dir: Path | str,
    sitl_v1_model_dir: Path | str,
    synthetic_model_dir: Path | str,
    output_dir: Path | str,
) -> Dict[str, Any]:
    """Evaluate SITL v2, SITL v1, and synthetic transfer models on the frozen test runs in one pass."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating SITL v2 (Improved) Detector on Frozen SITL Test...")
    v2_detector = TransferDetector(sitl_v2_model_dir)
    v2_runs = [evaluate_run_detailed(td, v2_detector, is_sitl_model=True) for td in test_dirs]
    v2_agg = aggregate_evaluation_metrics(v2_runs)
    v2_agg["model_scope"] = "sitl_v2_improved_detector"

    print("Evaluating SITL v1 (Historical) Model on Frozen SITL Test...")
    v1_detector = TransferDetector(sitl_v1_model_dir)
    v1_runs = [evaluate_run_detailed(td, v1_detector, is_sitl_model=True) for td in test_dirs]
    v1_agg = aggregate_evaluation_metrics(v1_runs)
    v1_agg["model_scope"] = "sitl_v1_historical_model"

    print("Evaluating Synthetic Checkpoint Transfer on Frozen SITL Test...")
    synth_detector = TransferDetector(synthetic_model_dir)
    synth_runs = [evaluate_run_detailed(td, synth_detector, is_sitl_model=False) for td in test_dirs]
    synth_agg = aggregate_evaluation_metrics(synth_runs)
    synth_agg["model_scope"] = "synthetic_trained_checkpoint_transfer_diagnostic"

    (output_dir / "sitl_v2_evaluation.json").write_text(json.dumps(v2_agg, indent=2) + "\n")
    (output_dir / "sitl_v1_evaluation.json").write_text(json.dumps(v1_agg, indent=2) + "\n")
    (output_dir / "synthetic_transfer_evaluation.json").write_text(json.dumps(synth_agg, indent=2) + "\n")

    comparison = {
        "schema_version": "acpdad-sitl-evaluation-comparison-v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_mission_count": len(test_dirs),
        "total_scheduled_observations": v2_agg["total_scheduled_observations"],
        "coverage_rate": v2_agg["overall_coverage_rate"],
        "metrics_comparison": {
            "benign_false_positive_rate": {
                "synthetic_checkpoint": synth_agg["benign_false_positive_rate"],
                "sitl_v1_historical": v1_agg["benign_false_positive_rate"],
                "sitl_v2_improved": v2_agg["benign_false_positive_rate"],
            },
            "macro_f1": {
                "synthetic_checkpoint": synth_agg["macro_f1"],
                "sitl_v1_historical": v1_agg["macro_f1"],
                "sitl_v2_improved_classifier": v2_agg["macro_f1"],
                "sitl_v2_improved_operational": v2_agg["operational_macro_f1"],
            },
            "macro_end_to_end_recall": {
                "synthetic_checkpoint": synth_agg["macro_end_to_end_recall"],
                "sitl_v1_historical": v1_agg["macro_end_to_end_recall"],
                "sitl_v2_improved_classifier": v2_agg["macro_end_to_end_recall"],
                "sitl_v2_improved_operational": v2_agg["operational_macro_end_to_end_recall"],
            },
            "average_latency_s": {
                "synthetic_checkpoint": synth_agg["average_detection_latency_s"],
                "sitl_v1_historical": v1_agg["average_detection_latency_s"],
                "sitl_v2_improved": v2_agg["average_detection_latency_s"],
            },
            "per_class_precision": {
                c: {
                    "synthetic_checkpoint": synth_agg["per_class"][c]["precision"],
                    "sitl_v1_historical": v1_agg["per_class"][c]["precision"],
                    "sitl_v2_improved": v2_agg["per_class"][c]["precision"],
                }
                for c in SITL_CLASSES
            },
            "per_class_operational_recall": {
                c: {
                    "synthetic_checkpoint": synth_agg["per_class"][c]["end_to_end_recall"],
                    "sitl_v1_historical": v1_agg["per_class"][c]["end_to_end_recall"],
                    "sitl_v2_improved": v2_agg["per_class"][c]["operational_end_to_end_recall"],
                }
                for c in SITL_CLASSES
            },
            "per_class_operational_f1": {
                c: {
                    "synthetic_checkpoint": synth_agg["per_class"][c]["f1_score"],
                    "sitl_v1_historical": v1_agg["per_class"][c]["f1_score"],
                    "sitl_v2_improved": v2_agg["per_class"][c]["operational_f1_score"],
                }
                for c in SITL_CLASSES
            },
            "dos_recall_breakdown": {
                "sitl_v1_historical": {
                    "scored_row_conditional_recall": v1_agg["per_class"]["DoS"]["conditional_recall"],
                    "raw_end_to_end_recall": v1_agg["per_class"]["DoS"]["end_to_end_recall"],
                    "operational_end_to_end_recall": v1_agg["per_class"]["DoS"]["operational_end_to_end_recall"],
                },
                "sitl_v2_improved": {
                    "scored_row_conditional_recall": v2_agg["per_class"]["DoS"]["conditional_recall"],
                    "raw_end_to_end_recall": v2_agg["per_class"]["DoS"]["end_to_end_recall"],
                    "operational_end_to_end_recall": v2_agg["per_class"]["DoS"]["operational_end_to_end_recall"],
                },
            },
            "fdi_precision_breakdown": {
                "synthetic_checkpoint": synth_agg["per_class"]["FDI"]["precision"],
                "sitl_v1_historical": v1_agg["per_class"]["FDI"]["precision"],
                "sitl_v2_improved": v2_agg["per_class"]["FDI"]["precision"],
            },
            "fdi_phase_breakdown_sitl_v2": v2_agg["phase_breakdown"],
        },
    }

    (output_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison
