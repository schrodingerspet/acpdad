"""Source-level and adapter checks that do not pretend to execute ArduPilot."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from ..simulation import BASE, TEMPORAL, W
from .adapter import FeatureAdapter, sitl_schema, UNSUPPORTED_FEATURES
from .detector import TransferDetector
from .scenarios import ScenarioPolicy, ScenarioUnavailable, ScenarioRelay


class _FakeMessage:
    def __init__(self, kind, sequence, time_boot_ms=1000, relative_alt=10000, raw=b"mavlink-frame", **fields):
        self.kind = kind
        self.sequence = int(sequence)
        self.time_boot_ms = int(time_boot_ms)
        self.relative_alt = int(relative_alt)
        self.roll = float(fields.get("roll", 0.01))
        self.pitch = float(fields.get("pitch", 0.02))
        self.yaw = float(fields.get("yaw", 0.03))
        self.vx = int(fields.get("vx", 20))
        self.vy = int(fields.get("vy", 10))
        self.vz = int(fields.get("vz", -5))
        self.alt = int(fields.get("alt", 12000))
        self.lat = int(fields.get("lat", 472000000))
        self.lon = int(fields.get("lon", 85400000))
        self.hdg = int(fields.get("hdg", 9000))
        self.raw = bytes(raw)

    def get_type(self):
        return self.kind

    def get_seq(self):
        return self.sequence

    def get_srcSystem(self):
        return 1

    def get_srcComponent(self):
        return 1

    def get_msgbuf(self):
        return self.raw


def run_checks(model_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def check(name, passed, evidence, issue=""):
        results.append({"item": name, "status": "PASS" if passed else "FAIL", "evidence": evidence, "issue": issue})

    schema = sitl_schema()
    check("SITL schema remains a separately named 22/88 mapping", schema["version"] == "acpdad-sitl-mavlink-v1" and len(schema["base_features"]) == 22 and len(schema["temporal_features"]) == 88, {"schema": schema["version"], "base_count": len(schema["base_features"]), "temporal_count": len(schema["temporal_features"]), "window": W})
    check("Wi-Fi-only evidence is explicitly unavailable", all(name in UNSUPPORTED_FEATURES for name in ("ap_changed", "association_duration_s", "management_frame")) and all(schema["feature_availability"][name].startswith("unsupported") for name in UNSUPPORTED_FEATURES), {"unsupported_features": list(UNSUPPORTED_FEATURES)})

    run_id = "selfcheck_run_a"
    adapter = FeatureAdapter(run_id, max_gap_s=2.0)
    detector = TransferDetector(model_dir)
    predictions = []
    base_rows = []
    arrival = 10_000_000_000
    sequence = 250
    source_ms = 4000
    for index in range(12):
        attitude = _FakeMessage("ATTITUDE", sequence, source_ms, raw=f"att-{index}".encode())
        attitude_audit, _ = adapter.consume(attitude, arrival - 1_000_000)
        sequence = (sequence + 1) % 256
        source_ms += 100
        position = _FakeMessage("GLOBAL_POSITION_INT", sequence, source_ms, 10000 + index * 10, raw=f"pos-{index}".encode())
        _, row = adapter.consume(position, arrival)
        sequence = (sequence + 1) % 256
        if row is not None:
            base_rows.append(row)
            predictions.append(detector.predict(adapter, row))
        arrival += 100_000_000
    check("MAVLink physical units and 22-feature order are emitted", len(base_rows) == 12 and all(name in base_rows[-1] for name in BASE) and base_rows[-1]["height_cm"] == 1011.0 and abs(base_rows[-1]["vx_mps"] - 0.2) < 1e-9, {"rows": len(base_rows), "feature_count": len(BASE), "height_cm": base_rows[-1].get("height_cm"), "vx_mps": base_rows[-1].get("vx_mps")})
    check("Received message ledger retains exact MAVLink frame bytes", attitude_audit["raw_frame_hex"] == b"att-11".hex(), {"frame_sha256": attitude_audit["frame_sha256"], "raw_frame_hex": attitude_audit["raw_frame_hex"]})
    check("Clock-domain-sensitive and Wi-Fi features remain NaN", all(math.isnan(float(base_rows[-1][name])) for name in UNSUPPORTED_FEATURES), {"unsupported_nan": list(UNSUPPORTED_FEATURES)})
    check("Fallback starts immediately and temporal model starts at W=10", predictions[0]["stage"] == "fallback_base" and predictions[8]["stage"] == "fallback_base" and predictions[9]["stage"] == "primary_temporal", {"first_stage": predictions[0]["stage"], "ninth_stage": predictions[8]["stage"], "tenth_stage": predictions[9]["stage"], "window": W})
    check("Transfer predictions contain five probabilities and no label fields", all(abs(sum(row[f"prob_{name}"] for name in schema["classes"]) - 1.0) < 1e-6 for row in predictions) and not any("Target_Label" in row or "scenario" in row for row in predictions), {"prediction_count": len(predictions), "classes": schema["classes"], "keys": list(predictions[-1])})

    malformed = _FakeMessage("BAD_DATA", sequence, raw=b"bad frame")
    bad_audit, bad_features = adapter.consume(malformed, arrival)
    check("Malformed MAVLink frames are recorded, excluded from model inputs, and reset temporal history", bad_audit["malformed"] and bad_features is None and adapter.malformed_count == 1 and len(adapter.temporal_window) == 0 and adapter.pending_reset_reason == "malformed_observation", {"audit": bad_audit, "feature_row_emitted": bad_features is not None, "history_length": len(adapter.temporal_window), "pending_reset_reason": adapter.pending_reset_reason})

    previous_source = source_ms + 500
    seq_reg_adapter = FeatureAdapter("sequence_check", max_gap_s=2.0)
    seq_reg_adapter.consume(_FakeMessage("GLOBAL_POSITION_INT", 20, previous_source, raw=b"first"), 1_000_000_000)
    _, regression_row = seq_reg_adapter.consume(_FakeMessage("GLOBAL_POSITION_INT", 19, previous_source - 200, raw=b"second"), 1_100_000_000)
    check("Source timestamp and MAVLink sequence regression are derived from received data", regression_row["source_regression"] == 1.0 and regression_row["sequence_regression"] == 1.0, {"source_regression": regression_row["source_regression"], "sequence_regression": regression_row["sequence_regression"]})
    _, gap_row = seq_reg_adapter.consume(_FakeMessage("GLOBAL_POSITION_INT", 21, previous_source + 100, raw=b"after-gap"), 4_000_000_000)
    check("Long delivery gap resets causal history", gap_row["reset_reason"] == "delivery_gap" and seq_reg_adapter.gap_resets == 1 and len(seq_reg_adapter.temporal_window) == 1, {"reset_reason": gap_row["reset_reason"], "history_length": len(seq_reg_adapter.temporal_window)})
    recovery_ready = []
    for index in range(1, W):
        _, recovered = seq_reg_adapter.consume(
            _FakeMessage("GLOBAL_POSITION_INT", 21 + index, previous_source + 100 + index * 100, raw=f"recover-{index}".encode()),
            4_000_000_000 + index * 100_000_000,
        )
        recovery_ready.append(bool(seq_reg_adapter.temporal_window and len(seq_reg_adapter.temporal_window) == W))
    check("Telemetry recovery rebuilds a fresh temporal window without crossing the gap", len(seq_reg_adapter.temporal_window) == W and not any(recovery_ready[:-1]) and recovery_ready[-1], {"rows_after_gap": len(recovery_ready), "window_size": len(seq_reg_adapter.temporal_window), "gap_resets": seq_reg_adapter.gap_resets})
    isolated = FeatureAdapter("selfcheck_run_b")
    check("Run state is isolated", isolated.run_id != adapter.run_id and isolated.message_count == 0 and isolated.last_position is None, {"run_a": adapter.run_id, "run_b": isolated.run_id})
    check("Prediction output preserves arrival ordering", all(int(predictions[index]["arrival_time_ns"]) < int(predictions[index + 1]["arrival_time_ns"]) for index in range(len(predictions) - 1)), {"observations": len(predictions)})

    policy = ScenarioPolicy("FDI", 30, 8, 12, {"scenarios": {"fdi": {"subtype": "step", "altitude_offset_m": 5.0, "drift_m_per_s": 0.25, "intermittent_period_s": 2.0}, "dos": {"drop_every_nth_global_position": 2}, "replay": {"delay_s": 2.0, "repetitions": 1, "buffer_s": 10.0}}})
    check("Attack labels use a half-open schedule interval", not policy.active(7.999) and policy.active(8.0) and policy.active(19.999) and not policy.active(20.0), {"interval_s": [policy.attack_start_s, policy.attack_start_s + policy.attack_duration_s], "boundary": "[start, end)"})
    try:
        ScenarioPolicy("Evil_Twin", 30, 8, 12, {"scenarios": {"fdi": {"subtype": "step", "altitude_offset_m": 1, "drift_m_per_s": 1, "intermittent_period_s": 1}, "dos": {"drop_every_nth_global_position": 2}, "replay": {"delay_s": 1, "repetitions": 1, "buffer_s": 2}}})
        evil_twin_rejected = False
    except ScenarioUnavailable as exc:
        evil_twin_rejected = "Wi-Fi" in str(exc) and "MAVLink" in str(exc)
    check("Evil_Twin is rejected unless a WLAN simulator is integrated", evil_twin_rejected, {"status": "unsupported", "reason": "MAVLink has no AP association semantics"})

    try:
        from .runner import DEFAULT_CONFIG, _load_config, _mavproxy_command, get_qgroundcontrol_port, _check_ports_available
        config = _load_config(DEFAULT_CONFIG)
        route = config["endpoints"]
        command_default = _mavproxy_command({"endpoints": route}, "mavproxy.py", "mavlink.tlog")
        route_ok = (
            route["controller"] == "udpin:127.0.0.1:14550"
            and "--out=udp:127.0.0.1:14550" in command_default
            and "--out=udp:127.0.0.1:14552" in command_default
            and not any("14600" in v for v in command_default)
            and all("127.0.0.1" in value for value in command_default if value.startswith("--master=") or value.startswith("--out="))
        )
        check("Controller and detector routes remain separated on loopback MAVProxy outputs", route_ok, {"controller": route["controller"], "mavproxy_outputs": [value for value in command_default if value.startswith("--out=")]})

        qgc_derived = get_qgroundcontrol_port(config)
        command_qgc = _mavproxy_command({"endpoints": route}, "mavproxy.py", "mavlink.tlog", qgc_port=qgc_derived)
        qgc_forward_ok = (
            f"--out=udp:127.0.0.1:{qgc_derived}" in command_qgc
            and "--out=udp:127.0.0.1:14550" in command_qgc
            and "--out=udp:127.0.0.1:14552" in command_qgc
            and qgc_derived == 14600
        )
        check("Optional QGroundControl observer tap forwards to dedicated UDP port without altering controller/relay routes", qgc_forward_ok, {"qgc_port": qgc_derived, "command_qgc_outputs": [v for v in command_qgc if v.startswith("--out=")]})

        inst1_config = dict(config, instance=1)
        inst2_config = dict(config, instance=2)
        inst1_port = get_qgroundcontrol_port(inst1_config)
        inst2_port = get_qgroundcontrol_port(inst2_config)
        override_port = get_qgroundcontrol_port(config, custom_port=14650)
        derivation_ok = inst1_port == 14601 and inst2_port == 14602 and override_port == 14650
        check("QGroundControl port derivation correctly offsets by instance and supports explicit override", derivation_ok, {"instance_0": qgc_derived, "instance_1": inst1_port, "instance_2": inst2_port, "override": override_port})

        collision_detected = False
        try:
            _check_ports_available(config, qgc_port=14550)
        except RuntimeError as exc:
            collision_detected = "collides with reserved SITL endpoint ports" in str(exc)
        check("QGroundControl port allocator rejects collisions with reserved SITL instance ports", collision_detected, {"rejected_port": 14550})
    except Exception as exc:
        check("Controller and detector routes remain separated on loopback MAVProxy outputs", False, f"{type(exc).__name__}: {exc}")

    pymavlink_status = "not_available"
    try:
        from pymavlink import mavutil
        from .scenarios import _parse_datagram
        wire = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        source = mavutil.mavlink.MAVLink_global_position_int_message(1000, 472000000, 85400000, 15000, 10000, 20, 10, -5, 9000)
        source_bytes = bytes(source.pack(wire))
        parsed = _parse_datagram(source_bytes, mavutil.mavlink)
        scenario_config = {"scenarios": {"fdi": {"subtype": "step", "altitude_offset_m": 5, "drift_m_per_s": 0.25, "intermittent_period_s": 2}, "dos": {"drop_every_nth_global_position": 2}, "replay": {"delay_s": 0.5, "repetitions": 1, "buffer_s": 10}}}
        policy = ScenarioPolicy("FDI", 4, 0, 3, scenario_config)
        relay = ScenarioRelay("wire_check", policy, scenario_config, mavutil.mavlink, [], [])
        relay.set_flight_start(0)
        changed, did_change = relay._rewrite_fdi(parsed, 1.0)
        changed_msg = _parse_datagram(changed, mavutil.mavlink)[0]
        wire_fdi_ok = did_change and changed_msg.alt == parsed[0].alt + 5000 and changed_msg.time_boot_ms == parsed[0].time_boot_ms and changed_msg.get_seq() == parsed[0].get_seq()
        check("FDI relay preserves MAVLink source time and packet sequence while changing defined telemetry", wire_fdi_ok, {"tested": True, "source_time_ms": changed_msg.time_boot_ms, "sequence": changed_msg.get_seq(), "altitude_offset_mm": changed_msg.alt - parsed[0].alt})
        dos_policy = ScenarioPolicy("DoS", 5, 0, 4, scenario_config)
        dos_relay = ScenarioRelay("dos_check", dos_policy, scenario_config, mavutil.mavlink, [], [])
        dos_relay.set_flight_start(0)
        first = dos_relay.process(source_bytes, 500_000_000)
        second = dos_relay.process(source_bytes, 600_000_000)
        check("DoS causes configured detector-branch loss without modifying source datagrams", len(first) == 1 and len(second) == 0 and dos_relay.dropped == 1, {"first_output_count": len(first), "second_output_count": len(second), "dropped": dos_relay.dropped})
        replay_policy = ScenarioPolicy("Replay", 5, 1, 3, scenario_config, replay_delay_s=0.5, replay_repetitions=1)
        replay_relay = ScenarioRelay("replay_check", replay_policy, scenario_config, mavutil.mavlink, [], [])
        replay_relay.set_flight_start(0)
        replay_relay.process(source_bytes, 200_000_000)
        later = mavutil.mavlink.MAVLink_global_position_int_message(1700, 472000100, 85400100, 19000, 14000, 30, 20, -4, 9100)
        later_bytes = bytes(later.pack(wire))
        replay_relay.process(later_bytes, 500_000_000)
        current = mavutil.mavlink.MAVLink_global_position_int_message(2100, 472000200, 85400200, 21000, 16000, 40, 30, -3, 9200)
        current_bytes = bytes(current.pack(wire))
        outputs = replay_relay.process(current_bytes, 2_000_000_000)
        replay_message = _parse_datagram(outputs[0], mavutil.mavlink)[0] if outputs else None
        check("Replay forwards an earlier exact MAVLink datagram with stale source time/sequence", outputs == [later_bytes] and replay_message is not None and replay_message.time_boot_ms == 1700 and replay_message.get_seq() == _parse_datagram(later_bytes, mavutil.mavlink)[0].get_seq() and replay_relay.replayed == 1, {"output_matches_captured_segment": outputs == [later_bytes], "replayed": replay_relay.replayed})
        malformed_relay = ScenarioRelay("malformed_check", ScenarioPolicy("benign", 5, 0, 0, scenario_config), scenario_config, mavutil.mavlink, [], [])
        malformed_output = malformed_relay.process(b"\x00not-a-mavlink-frame", 1_000_000_000)
        check("Malformed raw datagram is retained as invalid evidence rather than rewritten as telemetry", malformed_output == [b"\x00not-a-mavlink-frame"] and malformed_relay.events[-1]["message_type"] == "MALFORMED", {"forwarded_unchanged": malformed_output == [b"\x00not-a-mavlink-frame"], "message_type": malformed_relay.events[-1]["message_type"]})
        pymavlink_status = importlib_version = __import__("importlib.metadata", fromlist=["version"]).version("pymavlink")
    except ImportError:
        results.append({"item": "Wire-level MAVLink frame mutation/replay checks", "status": "NOT VERIFIED", "evidence": "pymavlink is not installed in this environment", "issue": "Install requirements-sitl.txt and rerun sitl selfcheck"})
    except Exception as exc:
        results.append({"item": "Wire-level MAVLink frame mutation/replay checks", "status": "FAIL", "evidence": f"{type(exc).__name__}: {exc}", "issue": "Inspect the pymavlink codec/relay path"})

    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": "ACPDAD SITL integration checks; fake-message checks are not an ArduPilot execution",
        "pymavlink": pymavlink_status,
        "checkpoint_sha256": hashlib.sha256((Path(model_dir) / "checkpoint.pkl").read_bytes()).hexdigest(),
        "items": results,
        "passed": sum(row["status"] == "PASS" for row in results),
        "failed": sum(row["status"] == "FAIL" for row in results),
        "not_verified": sum(row["status"] == "NOT VERIFIED" for row in results),
    }
    path = output_dir / f"sitl_selfcheck_{time.time_ns()}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    payload["report"] = str(path)
    return payload
