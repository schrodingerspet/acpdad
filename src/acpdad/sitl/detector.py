"""Streaming transfer inference for received SITL/MAVLink feature rows."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflow import load_checkpoint
from ..simulation import BASE, CLASSES, W, MAX_GAP_NS
from .adapter import FeatureAdapter


class TransferDetector:
    """Run the frozen synthetic checkpoint without fitting or tuning it."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        manifest_path = self.model_dir / "manifest.json"
        is_sitl = False
        manifest_data = {}
        if manifest_path.is_file():
            try:
                import json
                manifest_data = json.loads(manifest_path.read_text())
                if manifest_data.get("provenance") == "SITL_TRAINED_MODEL" or "sitl" in manifest_data.get("contract", ""):
                    is_sitl = True
            except Exception:
                pass
        if is_sitl:
            import hashlib
            import pickle
            ck = self.model_dir / "checkpoint.pkl"
            ck_hash = hashlib.sha256(ck.read_bytes()).hexdigest()
            if ck_hash != manifest_data["checkpoint_sha256"]:
                raise ValueError(f"SITL checkpoint digest mismatch: {ck_hash} != {manifest_data['checkpoint_sha256']}")
            with ck.open("rb") as f:
                self.checkpoint = pickle.load(f)
        else:
            self.checkpoint = load_checkpoint(self.model_dir)
        self.is_sitl = is_sitl
        self.models = self.checkpoint["models"]
        self.classes = list(self.checkpoint.get("classes", CLASSES))

    @staticmethod
    def _event_label(probabilities, threshold, classes):
        probabilities = np.asarray(probabilities, dtype=float)
        index = int(np.argmax(probabilities))
        if "Evil_Twin" in classes:
            evil_index = classes.index("Evil_Twin")
            if float(probabilities[evil_index]) > 0.0 and probabilities[evil_index] >= float(threshold):
                index = evil_index
        return classes[index], float(probabilities[index])

    def predict(self, adapter, row: dict) -> dict:
        """Predict from causal features only; labels and scenario metadata are absent."""
        base = np.asarray([row[name] for name in BASE], dtype=float)
        if row.get("temporal_ready"):
            model_name = "primary"
            model = self.models[model_name]
            vector = adapter.temporal_features()
            threshold = self.checkpoint["primary_threshold"]
            stage = "primary_temporal"
        else:
            selected = self.checkpoint.get("fallback_selected", "base")
            if selected == "short3" and adapter.short3_features() is not None:
                model_name = "fallback_short3"
                vector = adapter.short3_features()
                stage = "fallback_short3"
            else:
                model_name = "fallback_base"
                vector = base
                stage = "fallback_base"
            model = self.models[model_name]
            threshold = self.checkpoint["fallback_threshold"]
        vec_arr = np.asarray(vector, dtype=float).reshape(1, -1)
        if getattr(self, "is_sitl", False):
            vec_arr = np.nan_to_num(vec_arr, nan=0.0)
            if self.checkpoint.get("altitude_neutralized", False):
                vec_arr[:, 0] = 0.0
                if stage == "primary_temporal" and vec_arr.shape[1] > 22:
                    vec_arr[:, 22] = 0.0
        probabilities = model.predict_proba(vec_arr)[0]
        class_order = list(model.classes_)
        by_class = {str(name): (float(probabilities[class_order.index(name)]) if name in class_order else 0.0) for name in CLASSES}
        prediction, confidence = self._event_label([by_class[name] for name in CLASSES], threshold, CLASSES)
        residual = float(row.get("vertical_residual_m", math.nan))
        return {
            "run_id": row["run_id"],
            "observation_id": row["observation_id"],
            "arrival_time_ns": int(row["arrival_time_ns"]),
            "source_time_ms": int(row["source_time_ms"]),
            "sequence": int(row["sequence"]),
            "system_id": int(row["system_id"]),
            "component_id": int(row["component_id"]),
            "prediction": prediction,
            "confidence": confidence,
            "stage": stage,
            "health_status": f"RESET_{str(row.get('reset_reason')).upper()}_PARTIAL_FEATURES" if row.get("reset_reason") else "PARTIAL_FEATURES_UNSUPPORTED_FIELDS",
            "communication_warning": bool(row.get("reset_reason") == "delivery_gap"),
            "attack_alert_candidate": prediction != "Benign",
            "alert_policy_status": "NOT_CALIBRATED_FOR_SITL_TRANSFER",
            "physics_warning_candidate": bool(math.isfinite(residual) and residual > float(self.checkpoint["physics_threshold"])),
            **{f"prob_{name}": by_class[name] for name in CLASSES},
        }


def infer_saved_capture(run_dir, model_dir, output_path):
    """Re-run inference over saved features, in causal arrival order and without labels."""
    run_dir = Path(run_dir)
    source = run_dir / "features.csv"
    if not source.is_file():
        raise FileNotFoundError(f"SITL feature ledger not found: {source}")
    frame = pd.read_csv(source)
    forbidden = {"Target_Label", "scenario", "attack_active", "attack_schedule", "seed", "split"}
    found = sorted(forbidden & set(frame.columns))
    if found:
        raise ValueError(f"Ground-truth/experiment metadata cannot enter inference: {found}")
    required = {"run_id", "observation_id", "arrival_time_ns", "source_time_ms", "sequence", "system_id", "component_id", *BASE}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"SITL feature ledger is missing columns: {missing}")
    run_ids = frame["run_id"].astype(str).unique().tolist()
    if len(run_ids) > 1:
        raise ValueError(f"A saved capture must contain exactly one run; found {run_ids}")
    arrivals = pd.to_numeric(frame["arrival_time_ns"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arrivals).all() or np.any(np.diff(arrivals) < 0):
        raise ValueError("Saved capture rows must already be in nondecreasing MAVLink arrival order")
    detector = TransferDetector(model_dir)
    config_path = run_dir / "config.json"
    max_gap_ns = MAX_GAP_NS
    if config_path.is_file():
        import json
        config = json.loads(config_path.read_text())
        max_gap_ns = int(float(config["collection"]["max_gap_s"]) * 1e9)
    outputs = []
    current_run = None
    adapter = None
    last_arrival = None
    for row in frame.to_dict("records"):
        run_id = str(row["run_id"])
        arrival = int(row["arrival_time_ns"])
        if current_run != run_id:
            current_run = run_id
            adapter = FeatureAdapter(run_id)
            adapter.temporal_window = deque(maxlen=W)
            last_arrival = None
        if last_arrival is not None and arrival - last_arrival > max_gap_ns:
            adapter.temporal_window.clear()
            row["reset_reason"] = "delivery_gap"
        else:
            row["reset_reason"] = ""
        adapter.temporal_window.append(np.asarray([row[name] for name in BASE], dtype=float))
        row["temporal_ready"] = len(adapter.temporal_window) == W
        event = detector.predict(adapter, row)
        for metadata in ("datagram_index", "datagram_sha256"):
            if metadata in row:
                event[metadata] = row[metadata]
        outputs.append(event)
        last_arrival = arrival
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as stream:
        pd.DataFrame(outputs).to_csv(stream, index=False)
    if len(outputs) != len(frame):
        raise RuntimeError("Re-inference row count differs from selected SITL observations")
    return {
        "status": "completed",
        "run_id": str(frame.run_id.iloc[0]) if len(frame) else None,
        "rows": len(outputs),
        "output": str(output_path),
        "arrival_order_preserved": True,
        "ground_truth_used": False,
        "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
    }
