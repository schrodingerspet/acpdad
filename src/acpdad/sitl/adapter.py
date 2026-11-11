"""Causal mapping from received MAVLink messages into ACPDAD feature names.

The output column order matches the existing V5.4R2 checkpoint input order,
but unsupported Wi-Fi features are NaN and recorded in the SITL schema. The
adapter does not infer AP identity from a MAVLink system/component ID.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque

import numpy as np

from ..simulation import BASE, TEMPORAL, W
from . import SITL_SCHEMA_VERSION

UNSUPPORTED_FEATURES = (
    "packet_age_s",
    "ap_changed",
    "association_duration_s",
    "noninitial_association_duration_s",
    "packet_age_excess_s",
    "association_latency_excess_s",
    "management_frame",
)

FEATURE_AVAILABILITY = {
    "height_cm": "GLOBAL_POSITION_INT.relative_alt, millimetres converted to centimetres",
    "vx_mps": "GLOBAL_POSITION_INT.vx, centimetres/second converted to metres/second",
    "vy_mps": "GLOBAL_POSITION_INT.vy, centimetres/second converted to metres/second",
    "vz_down_mps": "GLOBAL_POSITION_INT.vz, positive down, centimetres/second converted to metres/second",
    "pitch_deg": "latest ATTITUDE.pitch converted from radians; unavailable if older than configured maximum age",
    "roll_deg": "latest ATTITUDE.roll converted from radians; unavailable if older than configured maximum age",
    "yaw_deg": "latest ATTITUDE.yaw converted from radians; unavailable if older than configured maximum age",
    "source_regression": "GLOBAL_POSITION_INT.time_boot_ms compared with the previous received position source time",
    "sequence_regression": "MAVLink packet sequence compared modulo 256 for the same system/component",
    "duplicate_payload": "SHA-256 of exact received MAVLink frame compared with the previous 30 received frames",
    "packet_age_s": "unsupported: host monotonic receive time and autopilot boot time are not aligned clocks",
    "arrival_delta_s": "monotonic receive-time difference between consecutive GLOBAL_POSITION_INT messages",
    "received_rate_pps": "all received MAVLink frames in the trailing one-second arrival-time window",
    "sequence_gap": "positive modulo-256 packet-sequence distance minus one, capped at 100",
    "ap_changed": "unsupported: MAVLink has no Wi-Fi AP/BSSID association field",
    "association_duration_s": "unsupported: MAVLink has no Wi-Fi association state",
    "noninitial_association_duration_s": "unsupported: MAVLink has no Wi-Fi association identity",
    "packet_age_excess_s": "unsupported: synchronized host/autopilot clocks are not established",
    "association_latency_excess_s": "unsupported: MAVLink has no Wi-Fi association transition",
    "frame_length": "byte length of the exact received MAVLink frame",
    "management_frame": "unsupported: MAVLink frames are not IEEE 802.11 management frames",
    "vertical_residual_m": "absolute relative-altitude change plus down-positive vertical speed integrated over positive source-time delta",
}


def sitl_schema() -> dict:
    """Return the source-specific mapping contract, distinct from synthetic data."""
    return {
        "version": SITL_SCHEMA_VERSION,
        "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
        "source": "MAVLink v2 received from a localhost-only MAVProxy route",
        "base_features": list(BASE),
        "temporal_features": list(TEMPORAL),
        "base_count": len(BASE),
        "temporal_count": len(TEMPORAL),
        "window": W,
        "classes": ["Benign", "DoS", "Replay", "Evil_Twin", "FDI"],
        "target": "Target_Label",
        "metadata": ["run_id", "observation_id", "arrival_time_ns", "source_time_ms", "sequence", "system_id", "component_id"],
        "model_input_policy": "Only ordered feature values from received MAVLink messages; metadata and scenario ground truth excluded",
        "temporal_transform": ["raw", "rolling mean", "rolling sample standard deviation (ddof=1)", "first difference"],
        "temporal_window": W,
        "temporal_reset": "new run, malformed or invalid required observation, or more than configured max_gap_s between received position messages",
        "transfer_only": True,
        "unsupported_features": list(UNSUPPORTED_FEATURES),
        "feature_availability": FEATURE_AVAILABILITY,
        "units": {
            "height_cm": "cm relative to home",
            "vx_mps": "m/s, north/east-frame X as supplied by GLOBAL_POSITION_INT",
            "vy_mps": "m/s, north/east-frame Y as supplied by GLOBAL_POSITION_INT",
            "vz_down_mps": "m/s, positive down",
            "pitch_deg": "degrees",
            "roll_deg": "degrees",
            "yaw_deg": "degrees",
            "arrival_delta_s": "s",
            "received_rate_pps": "MAVLink frames/s",
            "frame_length": "bytes",
            "vertical_residual_m": "m",
        },
        "differences_from_synthetic_contract": {
            "sequence_regression": "MAVLink's 8-bit packet sequence, modulo 256; synthetic packets used a 12-bit modeled sequence",
            "sequence_gap": "MAVLink packet sequence gap, modulo 256; this is not a Wi-Fi frame sequence estimate",
            "packet_age_s": "NaN because boot and host monotonic clocks are not synchronized",
            "wifi_fields": "NaN and explicitly unsupported; no AP identity, roaming, association, or management-frame evidence is synthesized",
        },
    }


def _field(message, name, default=math.nan):
    value = getattr(message, name, default)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def frame_bytes(message) -> bytes:
    getter = getattr(message, "get_msgbuf", None)
    if callable(getter):
        value = getter()
        if value:
            return bytes(value)
    return b""


class FeatureAdapter:
    """Build one ordered feature row per received GLOBAL_POSITION_INT."""

    def __init__(self, run_id: str, max_attitude_age_s: float = 1.0, rate_window_s: float = 1.0, max_gap_s: float = 2.0):
        self.run_id = str(run_id)
        self.max_attitude_age_ns = int(max_attitude_age_s * 1e9)
        self.rate_window_ns = int(rate_window_s * 1e9)
        self.max_gap_ns = int(max_gap_s * 1e9)
        self.message_count = 0
        self.position_count = 0
        self.arrivals = deque()
        self.recent_hashes = deque(maxlen=30)
        self.last_sequence = {}
        self.last_source_time_ms = None
        self.last_position_arrival_ns = None
        self.last_position = None
        self.last_attitude = None
        self.last_vfr_hud = None
        self.temporal_window = deque(maxlen=W)
        self.gap_resets = 0
        self.malformed_count = 0
        self.pending_reset_reason = ""

    def reset_on_invalid(self, reason="invalid_observation"):
        self.temporal_window.clear()
        self.last_position = None
        self.last_position_arrival_ns = None
        self.last_vfr_hud = None
        self.pending_reset_reason = str(reason)

    def consume(self, message, arrival_time_ns: int) -> tuple[dict, dict | None]:
        """Return a message audit record and (for position messages) base row."""
        arrival = int(arrival_time_ns)
        kind = str(message.get_type())
        raw = frame_bytes(message)
        digest = hashlib.sha256(raw).hexdigest() if raw else None
        system_id = int(message.get_srcSystem()) if hasattr(message, "get_srcSystem") else 0
        component_id = int(message.get_srcComponent()) if hasattr(message, "get_srcComponent") else 0
        sequence = int(message.get_seq()) if hasattr(message, "get_seq") else -1
        self.message_count += 1
        self.arrivals.append(arrival)
        while self.arrivals and arrival - self.arrivals[0] > self.rate_window_ns:
            self.arrivals.popleft()
        rate_elapsed_s = min(self.rate_window_ns, arrival - self.arrivals[0]) / 1e9 if self.arrivals else 0.0
        prior_sequence = self.last_sequence.get((system_id, component_id))
        sequence_distance = ((sequence - prior_sequence) % 256) if prior_sequence is not None and sequence >= 0 else None
        sequence_regression = float(sequence_distance == 0 or (sequence_distance is not None and sequence_distance >= 128)) if sequence_distance is not None else math.nan
        sequence_gap = float(min(100, max(0, sequence_distance - 1))) if sequence_distance is not None and 0 < sequence_distance < 128 else (0.0 if sequence_distance == 0 else math.nan)
        duplicate = float(digest in self.recent_hashes) if digest is not None else math.nan
        if sequence >= 0:
            self.last_sequence[(system_id, component_id)] = sequence
        if digest is not None:
            self.recent_hashes.append(digest)
        audit = {
            "run_id": self.run_id,
            "message_index": self.message_count - 1,
            "message_type": kind,
            "system_id": system_id,
            "component_id": component_id,
            "sequence": sequence,
            "arrival_time_ns": arrival,
            "frame_length": len(raw),
            "frame_sha256": digest,
            "raw_frame_hex": raw.hex() if raw else None,
            "malformed": kind == "BAD_DATA",
        }
        if kind == "BAD_DATA":
            self.malformed_count += 1
            audit["issue"] = "pymavlink_bad_data"
            self.reset_on_invalid("malformed_observation")
            return audit, None
        if kind == "ATTITUDE":
            self.last_attitude = (arrival, _field(message, "roll"), _field(message, "pitch"), _field(message, "yaw"))
            return audit, None
        if kind == "VFR_HUD":
            self.last_vfr_hud = (arrival, _field(message, "alt"), _field(message, "climb"))
            return audit, None
        if kind != "GLOBAL_POSITION_INT":
            return audit, None

        self.position_count += 1
        observation_id = f"{self.run_id}_{self.position_count:08d}"
        source_ms = int(_field(message, "time_boot_ms", 0.0))
        reset_reason = self.pending_reset_reason
        self.pending_reset_reason = ""
        if self.last_position_arrival_ns is not None and arrival - self.last_position_arrival_ns > self.max_gap_ns:
            self.temporal_window.clear()
            self.last_position = None
            self.gap_resets += 1
            reset_reason = "delivery_gap"
        prior_source = self.last_source_time_ms
        source_regression = float(prior_source is not None and source_ms <= prior_source)
        if self.last_attitude is not None and 0 <= arrival - self.last_attitude[0] <= self.max_attitude_age_ns:
            roll, pitch, yaw = self.last_attitude[1:]
        else:
            roll = pitch = yaw = math.nan
        alt_m = _field(message, "alt") / 1000.0
        relative_alt_m = _field(message, "relative_alt") / 1000.0
        vx = _field(message, "vx") / 100.0
        vy = _field(message, "vy") / 100.0
        vz_down = _field(message, "vz") / 100.0
        if not all(math.isfinite(value) for value in (relative_alt_m, vx, vy, vz_down)):
            audit["issue"] = "invalid_required_physical_fields"
            self.reset_on_invalid("invalid_features")
            return audit, None
        arrival_delta = (arrival - self.last_position_arrival_ns) / 1e9 if self.last_position_arrival_ns is not None and not reset_reason else math.nan
        source_delta = (source_ms - prior_source) / 1000.0 if prior_source is not None and source_ms > prior_source and not reset_reason else math.nan
        
        kinematic_residual = math.nan
        if self.last_position is not None and math.isfinite(source_delta) and source_delta > 0:
            previous_height_m = self.last_position["height_cm"] / 100.0
            kinematic_residual = abs(relative_alt_m - previous_height_m + vz_down * source_delta)

        sensor_residual = math.nan
        if self.last_vfr_hud is not None and 0 <= arrival - self.last_vfr_hud[0] <= self.max_attitude_age_ns:
            vfr_alt = self.last_vfr_hud[1]
            if math.isfinite(vfr_alt) and math.isfinite(alt_m):
                sensor_residual = abs(alt_m - vfr_alt)

        candidates = [c for c in (kinematic_residual, sensor_residual) if math.isfinite(c)]
        vertical_residual = max(candidates) if candidates else math.nan
        features = {name: math.nan for name in BASE}
        features.update({
            "height_cm": relative_alt_m * 100.0,
            "vx_mps": vx,
            "vy_mps": vy,
            "vz_down_mps": vz_down,
            "pitch_deg": math.degrees(pitch) if math.isfinite(pitch) else math.nan,
            "roll_deg": math.degrees(roll) if math.isfinite(roll) else math.nan,
            "yaw_deg": math.degrees(yaw) if math.isfinite(yaw) else math.nan,
            "source_regression": source_regression,
            "sequence_regression": sequence_regression,
            "duplicate_payload": duplicate,
            "arrival_delta_s": arrival_delta,
            "received_rate_pps": float(len(self.arrivals)) / rate_elapsed_s if rate_elapsed_s > 0 else math.nan,
            "sequence_gap": sequence_gap,
            "frame_length": float(len(raw)),
            "vertical_residual_m": vertical_residual,
        })
        row = {
            "run_id": self.run_id,
            "observation_id": observation_id,
            "arrival_time_ns": arrival,
            "source_time_ms": source_ms,
            "sequence": sequence,
            "system_id": system_id,
            "component_id": component_id,
            "reset_reason": reset_reason,
            **features,
        }
        self.temporal_window.append(np.asarray([features[x] for x in BASE], dtype=float))
        row["temporal_ready"] = len(self.temporal_window) == W
        self.last_source_time_ms = source_ms
        self.last_position_arrival_ns = arrival
        self.last_position = {"height_cm": features["height_cm"]}
        return audit, row

    def temporal_features(self):
        if len(self.temporal_window) < W:
            return None
        values = np.asarray(self.temporal_window, dtype=float)
        return np.concatenate((values[-1], values.mean(axis=0), values.std(axis=0, ddof=1), values[-1] - values[-2]))

    def short3_features(self):
        if len(self.temporal_window) < 3:
            return None
        values = np.asarray(list(self.temporal_window)[-3:], dtype=float)
        return np.concatenate((values[-1], values.mean(axis=0), values.std(axis=0, ddof=1), values[-1] - values[-2]))
