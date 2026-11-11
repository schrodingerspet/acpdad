"""Ordered localhost MAVLink reception, causal feature extraction and transfer inference."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import socket
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from ..simulation import BASE
from . import SITL_SCHEMA_VERSION
from .adapter import FeatureAdapter, sitl_schema
from .detector import TransferDetector
from .scenarios import ScenarioPolicy, ScenarioRelay, ground_truth_row


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DetectorReceiver(threading.Thread):
    def __init__(self, run_id, host, port, adapter, detector, messages, features, predictions, health_events, errors, mavlink_module, health_gap_s=2.0):
        super().__init__(name=f"acpdad-sitl-detector-{run_id}", daemon=True)
        self.run_id = run_id
        self.host = str(host)
        self.port = int(port)
        self.adapter = adapter
        self.detector = detector
        self.messages = messages
        self.features = features
        self.predictions = predictions
        self.health_events = health_events
        self.errors = errors
        self.mavlink = mavlink_module
        self.health_gap_ns = int(float(health_gap_s) * 1e9)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.socket = None
        self.datagram_index = 0
        self.started_ns = None
        self.last_datagram_ns = None
        self.last_position_ns = None
        self.stream_warning = False
        self.position_warning = False

    def _health(self, now_ns):
        if self.started_ns is None:
            return
        stream_age = (now_ns - (self.last_datagram_ns or self.started_ns)) / 1e9
        position_age = (now_ns - (self.last_position_ns or self.started_ns)) / 1e9
        stream_bad = stream_age * 1e9 > self.health_gap_ns
        position_bad = position_age * 1e9 > self.health_gap_ns
        if stream_bad != self.stream_warning:
            self.health_events.append({"run_id": self.run_id, "time_ns": int(now_ns), "event": "TELEMETRY_STREAM_GAP" if stream_bad else "TELEMETRY_STREAM_RECOVERED", "age_s": stream_age, "attack_classification": False})
            self.stream_warning = stream_bad
        if position_bad != self.position_warning:
            self.health_events.append({"run_id": self.run_id, "time_ns": int(now_ns), "event": "POSITION_TELEMETRY_MISSING" if position_bad else "POSITION_TELEMETRY_RECOVERED", "age_s": position_age, "attack_classification": False})
            self.position_warning = position_bad

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(0.25)
        self.socket = sock
        self.started_ns = time.monotonic_ns()
        self.ready.set()
        parser = self.mavlink.MAVLink(None)
        try:
            while not self.stop_event.is_set():
                try:
                    datagram, address = sock.recvfrom(65535)
                except socket.timeout:
                    self._health(time.monotonic_ns())
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise
                arrival_ns = time.monotonic_ns()
                self.last_datagram_ns = arrival_ns
                if not address or not str(address[0]).startswith("127."):
                    self.errors.append(f"Rejected non-loopback detector datagram from {address!r}")
                    continue
                datagram_index = self.datagram_index
                self.datagram_index += 1
                datagram_sha = _sha(datagram)
                parsed = False
                for value in datagram:
                    try:
                        message = parser.parse_char(bytes((value,)))
                    except Exception as exc:
                        self.errors.append(f"Malformed MAVLink datagram {datagram_index}: {type(exc).__name__}: {exc}")
                        continue
                    if message is None:
                        continue
                    parsed = True
                    audit, row = self.adapter.consume(message, arrival_ns)
                    audit["datagram_index"] = datagram_index
                    audit["datagram_sha256"] = datagram_sha
                    audit["raw_frame_hex"] = bytes(message.get_msgbuf() or b"").hex()
                    self.messages.append(audit)
                    if audit["message_type"] == "GLOBAL_POSITION_INT" and not audit["malformed"]:
                        self.last_position_ns = arrival_ns
                    if row is None:
                        continue
                    row["datagram_index"] = datagram_index
                    row["datagram_sha256"] = datagram_sha
                    row["raw_frame_hex"] = bytes(message.get_msgbuf() or b"").hex()
                    self.features.append(row)
                    try:
                        prediction = self.detector.predict(self.adapter, row)
                        prediction["datagram_index"] = datagram_index
                        prediction["datagram_sha256"] = datagram_sha
                        self.predictions.append(prediction)
                    except Exception as exc:
                        self.errors.append(f"Inference failed at {row['observation_id']}: {type(exc).__name__}: {exc}")
                        self.predictions.append({
                            "run_id": self.run_id,
                            "observation_id": row["observation_id"],
                            "arrival_time_ns": int(arrival_ns),
                            "source_time_ms": int(row["source_time_ms"]),
                            "sequence": int(row["sequence"]),
                            "prediction": "INFERENCE_ERROR",
                            "confidence": math.nan,
                            "stage": "inference_error",
                            "health_status": "INFERENCE_ERROR",
                            "attack_alert_candidate": False,
                            "datagram_index": datagram_index,
                            "datagram_sha256": datagram_sha,
                        })
                self._health(time.monotonic_ns())
                if not parsed:
                    self.errors.append(f"No valid MAVLink frame decoded from datagram {datagram_index} ({datagram_sha})")
                    self.adapter.reset_on_invalid("malformed_datagram")
        except Exception as exc:
            self.errors.append(f"Detector receiver stopped: {type(exc).__name__}: {exc}")
        finally:
            sock.close()

    def stop(self):
        self.stop_event.set()
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass


class CaptureContext:
    """Run one relay/collector pair and persist a label-free prediction ledger."""

    def __init__(self, run_id, output_dir, config, model_dir, policy, mavlink_module):
        self.run_id = str(run_id)
        self.output_dir = Path(output_dir)
        self.config = config
        self.policy = policy
        self.mavlink = mavlink_module
        self.model_dir = Path(model_dir)
        self.detector = TransferDetector(self.model_dir)
        self.adapter = FeatureAdapter(
            self.run_id,
            max_attitude_age_s=config["collection"]["max_attitude_age_s"],
            rate_window_s=config["collection"]["rate_window_s"],
            max_gap_s=config["collection"]["max_gap_s"],
        )
        self.messages = []
        self.features = []
        self.predictions = []
        self.health_events = []
        self.relay_events = []
        self.errors = []
        endpoint = config["endpoints"]
        self.receiver = DetectorReceiver(
            self.run_id, endpoint["detector_bind_host"], endpoint["detector_bind_port"],
            self.adapter, self.detector, self.messages, self.features, self.predictions,
            self.health_events, self.errors, mavlink_module, config["collection"]["max_gap_s"],
        )
        self.relay = ScenarioRelay(self.run_id, policy, config, mavlink_module, self.relay_events, self.errors)
        self.start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def start(self, flight_start_ns=None, timeout_s=5.0, anchor_now=True):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.receiver.start()
        if not self.receiver.ready.wait(timeout_s):
            raise RuntimeError("Could not bind the loopback detector MAVLink UDP endpoint")
        self.relay.start()
        if not self.relay.ready.wait(timeout_s):
            raise RuntimeError("Could not bind the loopback MAVLink scenario relay endpoint")
        if flight_start_ns is not None:
            self.relay.set_flight_start(flight_start_ns)
        elif anchor_now:
            self.relay.set_flight_start(time.monotonic_ns())
        self.started = True

    def set_flight_start(self, monotonic_ns=None):
        self.relay.set_flight_start(monotonic_ns)

    def get_live_state(self) -> dict:
        now_ns = time.monotonic_ns()
        relay_state = self.relay.get_live_state() if hasattr(self.relay, "get_live_state") else {}
        latest_pred = self.predictions[-1] if self.predictions else None
        latest_feat = self.features[-1] if self.features else None
        pred_age_ms = ((now_ns - latest_pred["arrival_time_ns"]) / 1e6) if latest_pred and latest_pred.get("arrival_time_ns") else None

        v_res = latest_feat.get("vertical_residual_m") if latest_feat else None
        p_thresh = float(self.detector.checkpoint.get("physics_threshold", 0.0108))
        phys_warning = bool(v_res is not None and math.isfinite(v_res) and v_res > p_thresh)

        probabilities = {}
        if latest_pred:
            from ..simulation import CLASSES
            for cls_name in CLASSES:
                probabilities[cls_name] = latest_pred.get(f"prob_{cls_name}", 0.0)

        recent_arr = getattr(self.adapter, "arrivals", None)
        rate_window_ns = getattr(self.adapter, "rate_window_ns", 1_000_000_000)
        rate_hz = 0.0
        if recent_arr:
            rate_elapsed = min(rate_window_ns, now_ns - recent_arr[0]) / 1e9 if recent_arr else 0.0
            if rate_elapsed > 0:
                rate_hz = round(float(len(recent_arr)) / rate_elapsed, 1)

        return {
            "run_id": self.run_id,
            "relay": relay_state,
            "ml_inference": {
                "latest_prediction": latest_pred.get("prediction") if latest_pred else "INITIALIZING",
                "confidence": latest_pred.get("confidence") if latest_pred else None,
                "stage": latest_pred.get("stage") if latest_pred else "awaiting_telemetry",
                "prediction_age_ms": round(pred_age_ms, 1) if pred_age_ms is not None else None,
                "is_stale": (pred_age_ms is not None and pred_age_ms > 1000.0) or (latest_pred is None),
                "probabilities": probabilities,
                "attack_alert_candidate": latest_pred.get("attack_alert_candidate", False) if latest_pred else False,
            },
            "detector_health": {
                "alert_active": latest_pred.get("attack_alert_candidate", False) if latest_pred else False,
                "physics_warning_active": phys_warning,
                "vertical_residual_m": round(v_res, 4) if v_res is not None and math.isfinite(v_res) else None,
                "vertical_residual_threshold_m": p_thresh,
                "watchdog_gap_active": getattr(self.receiver, "position_warning", False),
                "telemetry_stream_gap": getattr(self.receiver, "stream_warning", False),
                "datagrams_received": len(self.messages),
                "position_frames": len(self.features),
                "predictions_count": len(self.predictions),
                "forwarded": self.relay.forwarded,
                "dropped": self.relay.dropped,
                "mutated": self.relay.mutated,
                "replayed": self.relay.replayed,
                "telemetry_rate_hz": rate_hz,
            },
            "phases": self.policy.phases_summary() if hasattr(self.policy, "phases_summary") else [],
        }

    def stop(self):
        self.relay.stop()
        self.receiver.stop()
        self.relay.join(timeout=3.0)
        self.receiver.join(timeout=3.0)
        self.write_outputs()

    def write_outputs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        schema = sitl_schema()
        (self.output_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
        (self.output_dir / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")
        self._write_jsonl("received_messages.jsonl", self.messages)
        self._write_jsonl("relay_events.jsonl", self.relay_events)
        self._write_jsonl("health_events.jsonl", self.health_events)
        self._write_jsonl("runtime_errors.jsonl", [{"error": x} for x in self.errors])
        feature_columns = [
            "run_id", "observation_id", "arrival_time_ns", "source_time_ms", "sequence", "system_id", "component_id",
            "reset_reason", "datagram_index", "datagram_sha256", "raw_frame_hex", *BASE, "temporal_ready",
        ]
        feature_frame = pd.DataFrame(self.features)
        if feature_frame.empty:
            feature_frame = pd.DataFrame(columns=feature_columns)
        else:
            feature_frame = feature_frame.reindex(columns=feature_columns)
        feature_frame.to_csv(self.output_dir / "features.csv", index=False)
        prediction_frame = pd.DataFrame(self.predictions)
        if prediction_frame.empty:
            prediction_frame = pd.DataFrame(columns=["run_id", "observation_id", "arrival_time_ns", "prediction", "confidence", "stage"])
        prediction_frame.to_csv(self.output_dir / "predictions_label_free.csv", index=False)
        relay_frame = pd.DataFrame(self.relay_events)
        self._write_post_inference_truth(relay_frame, prediction_frame, feature_frame)
        self._write_jsonl("runtime_errors.jsonl", [{"error": x} for x in self.errors])
        (self.output_dir / "run_manifest.json").write_text(json.dumps({
            "run_id": self.run_id,
            "schema_version": SITL_SCHEMA_VERSION,
            "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
            "scenario": self.policy.name,
            "started_utc": self.start_utc,
            "scheduled_attack": {
                "start_s": self.policy.attack_start_s if self.policy.name != "benign" else None,
                "duration_s": self.policy.attack_duration_s if self.policy.name != "benign" else 0,
                "subtype": self.policy.fdi_subtype if self.policy.name == "FDI" else ("stale_packet_replay" if self.policy.name == "Replay" else None),
            },
            "sequence_phases": self.policy.phases_summary() if hasattr(self.policy, "phases_summary") else None,
            "detector_inputs": "received MAVLink payload and arrival metadata only",
            "ground_truth_joined_after_inference": True,
            "message_count": len(self.messages),
            "position_observation_count": len(self.features),
            "prediction_count": len(self.predictions),
            "health_event_count": len(self.health_events),
            "unsupported_features": schema["unsupported_features"],
            "malformed_message_count": self.adapter.malformed_count,
            "temporal_gap_resets": self.adapter.gap_resets,
            "relay_counts": {"forwarded": self.relay.forwarded, "dropped": self.relay.dropped, "mutated": self.relay.mutated, "replayed": self.relay.replayed},
            "transfer_model_checkpoint_sha256": self._checkpoint_sha256(),
            "synthetic_checkpoint_metrics_are_not_sitl_metrics": True,
            "errors": list(self.errors),
        }, indent=2, allow_nan=False) + "\n")

    def _checkpoint_sha256(self):
        path = self.model_dir / "checkpoint.pkl"
        return _sha(path.read_bytes()) if path.is_file() else None

    def _write_jsonl(self, name, items):
        path = self.output_dir / name
        with path.open("w", encoding="utf-8") as stream:
            for item in items:
                stream.write(json.dumps(item, separators=(",", ":"), allow_nan=False, default=self._json_default) + "\n")

    @staticmethod
    def _json_default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value) if np.isfinite(value) else None
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    def _write_post_inference_truth(self, relay_frame, prediction_frame, feature_frame):
        emitted = relay_frame[relay_frame.get("message_type", pd.Series(dtype=str)).eq("GLOBAL_POSITION_INT")].copy() if not relay_frame.empty else pd.DataFrame()
        if emitted.empty:
            pd.DataFrame(columns=["run_id", "event_index", "Target_Label", "prediction", "coverage"]).to_csv(self.output_dir / "ground_truth_joined_after_inference.csv", index=False)
            (self.output_dir / "metrics.json").write_text(json.dumps({"status": "no_position_messages_received", "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION"}, indent=2) + "\n")
            return
        prediction_by_hash = defaultdict(deque)
        if not feature_frame.empty and not prediction_frame.empty:
            predicted_by_id = prediction_frame.set_index("observation_id").to_dict("index")
            for feature in feature_frame.to_dict("records"):
                event = predicted_by_id.get(feature["observation_id"])
                digest = feature.get("datagram_sha256")
                if event is not None and isinstance(digest, str):
                    prediction_by_hash[digest].append(event)
        timeline = []
        for event in emitted.sort_values("event_index", kind="stable").to_dict("records"):
            elapsed = event.get("mission_elapsed_s")
            if event.get("attack_active") and event.get("scheduled_scenario") in ("combined", "concurrent"):
                label = "Combined_Attack"
            elif event.get("attack_active") and event.get("scheduled_scenario") != "benign":
                label = event.get("scheduled_scenario")
            elif elapsed is not None and hasattr(self.policy, "target_label"):
                label = self.policy.target_label(float(elapsed))
            elif elapsed is not None and self.policy.active(float(elapsed)):
                label = self.policy.name
            else:
                label = "Benign"
            outputs = event.get("outputs") or []
            if isinstance(outputs, str):
                try:
                    outputs = json.loads(outputs)
                except ValueError:
                    outputs = []
            assigned = None
            for output in outputs:
                if not isinstance(output, dict):
                    continue
                candidates = prediction_by_hash.get(output.get("sha256"))
                if candidates:
                    assigned = candidates.popleft()
                    break
            predicted = str(assigned["prediction"]) if assigned else "UNAVAILABLE"
            timeline.append({
                "run_id": self.run_id,
                "event_index": int(event["event_index"]),
                "mission_elapsed_s": elapsed,
                "Target_Label": label,
                "prediction": predicted,
                "confidence": assigned.get("confidence") if assigned else None,
                "coverage": assigned is not None,
                "relay_effect": event.get("effect"),
                "source_time_ms": event.get("source_time_ms"),
                "sequence": event.get("sequence"),
            })
        timeline_frame = pd.DataFrame(timeline)
        timeline_frame.to_csv(self.output_dir / "ground_truth_joined_after_inference.csv", index=False)
        labels = ["Benign", "DoS", "Replay", "Evil_Twin", "FDI"]
        for item in timeline:
            if item["Target_Label"] not in labels:
                labels.append(item["Target_Label"])
        matrix = {actual: {predicted: 0 for predicted in labels + ["UNAVAILABLE", "INFERENCE_ERROR"]} for actual in labels}
        for item in timeline:
            guess = item["prediction"]
            if guess not in matrix[item["Target_Label"]]:
                guess = "INFERENCE_ERROR"
            matrix[item["Target_Label"]][guess] += 1
        per_class = {}
        for label in labels:
            tp = matrix[label][label]
            support = sum(matrix[label].values())
            predicted_count = sum(matrix[actual][label] for actual in labels)
            precision = tp / predicted_count if predicted_count else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        evaluated = len(timeline)
        correctly_predicted = sum(matrix[label][label] for label in labels)
        benign_rows = [item for item in timeline if item["Target_Label"] == "Benign" and item["coverage"]]
        benign_false_positives = sum(item["prediction"] not in ("Benign", "UNAVAILABLE", "INFERENCE_ERROR") for item in benign_rows)
        (self.output_dir / "metrics.json").write_text(json.dumps({
            "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
            "evaluation_scope": "single transfer run; scheduled position observations emitted by MAVProxy, unavailable deliveries counted as misses",
            "accuracy_end_to_end": correctly_predicted / evaluated if evaluated else 0.0,
            "prediction_coverage": sum(item["coverage"] for item in timeline) / evaluated if evaluated else 0.0,
            "benign_scored_row_false_positive_rate": benign_false_positives / len(benign_rows) if benign_rows else None,
            "benign_scored_rows": len(benign_rows),
            "macro_f1_end_to_end": sum(item["f1"] for item in per_class.values()) / len(labels),
            "per_class": per_class,
            "confusion_matrix_including_unavailable": matrix,
            "predictions_are_frozen_synthetic_checkpoint_transfer_only": True,
            "not_a_sitl_trained_model_or_population_performance_estimate": True,
        }, indent=2, allow_nan=False) + "\n")

