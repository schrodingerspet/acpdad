"""Loopback-only MAVLink relay scenarios for detector-facing telemetry.

These scenarios alter only the MAVLink branch forwarded to the detector. The
controller-facing MAVProxy output remains the unmodified SITL stream. Labels
and attack schedules are written to sidecar files and never sent to inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import threading
import time
from collections import deque

SUPPORTED_SCENARIOS = ("benign", "FDI", "DoS", "Replay", "sequence", "combined", "concurrent")


class ScenarioUnavailable(ValueError):
    pass


def validate_scenario(name: str) -> str:
    if name == "Evil_Twin":
        raise ScenarioUnavailable(
            "Evil_Twin is not implemented in the SITL/MAVLink path: MAVLink has no Wi-Fi AP/BSSID or association semantics. "
            "Use a separately isolated WLAN simulator before making an Evil_Twin claim."
        )
    if name not in SUPPORTED_SCENARIOS:
        raise ScenarioUnavailable(f"Unsupported SITL scenario {name!r}; available: {', '.join(SUPPORTED_SCENARIOS)}")
    return name


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_datagram(data: bytes, mavlink_module):
    parser = mavlink_module.MAVLink(None)
    messages = []
    for value in data:
        message = parser.parse_char(bytes((value,)))
        if message is not None:
            messages.append(message)
    return messages


def _message_type(message):
    try:
        return str(message.get_type())
    except Exception:
        return "UNKNOWN"


def _message_frame(message):
    try:
        value = message.get_msgbuf()
        return bytes(value) if value else b""
    except Exception:
        return b""


def _source_fields(messages):
    for message in messages:
        if _message_type(message) == "GLOBAL_POSITION_INT":
            return int(getattr(message, "time_boot_ms", -1)), int(message.get_seq()), "GLOBAL_POSITION_INT"
    return -1, -1, _message_type(messages[0]) if messages else "MALFORMED"


class SequencePhase:
    """A distinct temporal phase in a multi-attack continuous flight mission."""

    def __init__(
        self,
        phase_index: int,
        name: str,
        scenario_type: str,
        duration_s: float,
        start_s: float,
        description: str = "",
        fdi_subtype: str = "step",
        fdi_offset_m: float = 5.0,
        dos_drop_every: int = 2,
        replay_delay_s: float = 5.0,
        replay_repetitions: int = 1,
    ):
        self.phase_index = int(phase_index)
        self.name = str(name)
        self.scenario_type = str(scenario_type)
        self.duration_s = float(duration_s)
        self.start_s = float(start_s)
        self.end_s = float(start_s + duration_s)
        self.description = str(description)
        self.fdi_subtype = str(fdi_subtype)
        self.fdi_offset_m = float(fdi_offset_m)
        self.dos_drop_every = int(dos_drop_every)
        self.replay_delay_s = float(replay_delay_s)
        self.replay_repetitions = int(replay_repetitions)

    def contains(self, elapsed_s: float | None) -> bool:
        if elapsed_s is None:
            return False
        return self.start_s <= elapsed_s < self.end_s

    def to_dict(self) -> dict:
        return {
            "phase_index": self.phase_index,
            "name": self.name,
            "scenario_type": self.scenario_type,
            "start_s": round(self.start_s, 2),
            "duration_s": round(self.duration_s, 2),
            "end_s": round(self.end_s, 2),
            "description": self.description,
            "is_attack": self.scenario_type != "benign",
            "params": {
                "fdi_subtype": self.fdi_subtype if self.scenario_type == "FDI" else None,
                "fdi_offset_m": self.fdi_offset_m if self.scenario_type == "FDI" else None,
                "dos_drop_every": self.dos_drop_every if self.scenario_type == "DoS" else None,
                "replay_delay_s": self.replay_delay_s if self.scenario_type == "Replay" else None,
                "replay_repetitions": self.replay_repetitions if self.scenario_type == "Replay" else None,
            },
        }


DEFAULT_SEQUENCE_PHASES = [
    {"name": "baseline", "scenario": "benign", "duration_s": 15.0, "desc": "Benign airborne baseline"},
    {"name": "fdi_injection", "scenario": "FDI", "duration_s": 20.0, "desc": "FDI step altitude injection (+5.0m)"},
    {"name": "recovery_fdi", "scenario": "benign", "duration_s": 15.0, "desc": "Benign recovery post-FDI"},
    {"name": "dos_injection", "scenario": "DoS", "duration_s": 20.0, "desc": "DoS packet drop (every 2nd position)"},
    {"name": "recovery_dos", "scenario": "benign", "duration_s": 15.0, "desc": "Benign recovery post-DoS"},
    {"name": "replay_injection", "scenario": "Replay", "duration_s": 20.0, "desc": "Replay of 5.0s-stale telemetry"},
    {"name": "final_recovery", "scenario": "benign", "duration_s": 15.0, "desc": "Final benign recovery before return and landing"},
]


class SequenceScenarioPolicy:
    """Manages sequential attack and recovery phases during one continuous flight."""

    def __init__(
        self,
        name: str = "sequence",
        duration_s: float = 120.0,
        attack_start_s: float | None = None,
        attack_duration_s: float | None = None,
        config: dict | None = None,
        fdi_subtype: str | None = None,
        replay_delay_s: float | None = None,
        replay_repetitions: int | None = None,
        fdi_offset_m: float | None = None,
        dos_drop_every: int | None = None,
        phase_specs: list[dict] | None = None,
    ):
        self.name = "sequence"
        self.config = config or {}
        specs = phase_specs or DEFAULT_SEQUENCE_PHASES
        fdi_cfg = self.config.get("scenarios", {}).get("fdi", {})
        dos_cfg = self.config.get("scenarios", {}).get("dos", {})
        replay_cfg = self.config.get("scenarios", {}).get("replay", {})

        offset_m = float(fdi_offset_m if fdi_offset_m is not None else fdi_cfg.get("altitude_offset_m", 5.0))
        subtype = str(fdi_subtype or fdi_cfg.get("subtype", "step"))
        drop_every = int(dos_drop_every if dos_drop_every is not None else dos_cfg.get("drop_every_nth_global_position", 2))
        delay_s = float(replay_delay_s if replay_delay_s is not None else replay_cfg.get("delay_s", 5.0))
        reps = int(replay_repetitions if replay_repetitions is not None else replay_cfg.get("repetitions", 1))

        self.phases: list[SequencePhase] = []
        cur_start = 0.0
        for idx, spec in enumerate(specs, start=1):
            dur = float(spec.get("duration_s", 15.0))
            scen = str(spec.get("scenario", "benign"))
            phase = SequencePhase(
                phase_index=idx,
                name=spec.get("name", f"phase_{idx}"),
                scenario_type=scen,
                duration_s=dur,
                start_s=cur_start,
                description=spec.get("desc", ""),
                fdi_subtype=subtype,
                fdi_offset_m=offset_m,
                dos_drop_every=drop_every,
                replay_delay_s=delay_s,
                replay_repetitions=reps,
            )
            self.phases.append(phase)
            cur_start += dur

        self.duration_s = max(float(duration_s), cur_start)
        self.attack_start_s = self.phases[1].start_s if len(self.phases) > 1 else 15.0
        self.attack_duration_s = cur_start - self.attack_start_s
        self.fdi_subtype = subtype
        self.replay_delay_s = delay_s
        self.replay_repetitions = reps
        self.fdi_offset = offset_m
        self.dos_drop_every = drop_every

    def active_phase(self, elapsed_s: float | None) -> SequencePhase | None:
        if elapsed_s is None:
            return None
        for phase in self.phases:
            if phase.contains(elapsed_s):
                return phase
        return None

    def active(self, elapsed_s: float | None) -> bool:
        phase = self.active_phase(elapsed_s)
        return phase is not None and phase.scenario_type != "benign"

    def active_scenario(self, elapsed_s: float | None) -> str:
        phase = self.active_phase(elapsed_s)
        return phase.scenario_type if phase is not None else "benign"

    def target_label(self, elapsed_s: float | None) -> str:
        phase = self.active_phase(elapsed_s)
        return phase.scenario_type if (phase is not None and phase.scenario_type != "benign") else "Benign"

    def fdi_offset_m(self, elapsed_s: float) -> float:
        phase = self.active_phase(elapsed_s)
        if phase is None or phase.scenario_type != "FDI":
            return 0.0
        attack_elapsed = max(0.0, elapsed_s - phase.start_s)
        if phase.fdi_subtype in ("step", "coordinated"):
            return float(phase.fdi_offset_m)
        if phase.fdi_subtype == "drift":
            drift_rate = float(self.config.get("scenarios", {}).get("fdi", {}).get("drift_m_per_s", 0.25))
            return drift_rate * attack_elapsed
        period = float(self.config.get("scenarios", {}).get("fdi", {}).get("intermittent_period_s", 2.0))
        return float(phase.fdi_offset_m) if int(attack_elapsed / period) % 2 == 0 else 0.0

    def phases_summary(self) -> list[dict]:
        return [p.to_dict() for p in self.phases]


class CombinedScenarioPolicy:
    """Manages concurrent FDI, DoS, and Replay injection during a single 2-lap flight."""

    def __init__(
        self,
        name: str = "combined",
        duration_s: float = 160.0,
        attack_start_s: float | None = None,
        attack_duration_s: float | None = None,
        config: dict | None = None,
        fdi_subtype: str | None = None,
        replay_delay_s: float | None = None,
        replay_repetitions: int | None = None,
        fdi_offset_m: float | None = None,
        dos_drop_every: int | None = None,
    ):
        self.name = str(name)
        self.config = config or {}
        fdi_cfg = self.config.get("scenarios", {}).get("fdi", {})
        dos_cfg = self.config.get("scenarios", {}).get("dos", {})
        replay_cfg = self.config.get("scenarios", {}).get("replay", {})

        self.attack_start_s = float(attack_start_s if attack_start_s is not None else 5.0)
        self.duration_s = float(duration_s if duration_s is not None else 160.0)
        self.attack_duration_s = float(attack_duration_s if attack_duration_s is not None else (self.duration_s - self.attack_start_s))
        self.fdi_subtype = str(fdi_subtype or fdi_cfg.get("subtype", "step"))
        self.fdi_offset = float(fdi_offset_m if fdi_offset_m is not None else fdi_cfg.get("altitude_offset_m", 5.0))
        self.dos_drop_every = int(dos_drop_every if dos_drop_every is not None else 3)
        self.replay_delay_s = float(replay_delay_s if replay_delay_s is not None else replay_cfg.get("delay_s", 5.0))
        self.injections_stopped = False
        self.attack_end_s = None

    def stop_injections(self, elapsed_s: float | None = None):
        self.injections_stopped = True
        if elapsed_s is not None:
            self.attack_end_s = float(elapsed_s)

    def active(self, elapsed_s: float | None) -> bool:
        if elapsed_s is None:
            return False
        if self.attack_end_s is not None:
            return self.attack_start_s <= elapsed_s < self.attack_end_s
        if self.injections_stopped:
            return False
        return elapsed_s >= self.attack_start_s

    def active_scenario(self, elapsed_s: float | None) -> str:
        return self.name if self.active(elapsed_s) else "benign"

    def target_label(self, elapsed_s: float | None) -> str:
        return "Combined_Attack" if self.active(elapsed_s) else "Benign"

    def fdi_offset_m(self, elapsed_s: float) -> float:
        if not self.active(elapsed_s):
            return 0.0
        return float(self.fdi_offset)

    def active_phase(self, elapsed_s: float | None):
        return None

    def phases_summary(self) -> list[dict]:
        return [
            {
                "phase_index": 1,
                "name": "baseline",
                "scenario_type": "benign",
                "start_s": 0.0,
                "duration_s": self.attack_start_s,
                "end_s": self.attack_start_s,
                "description": "Benign airborne baseline before concurrent attack",
                "is_attack": False,
            },
            {
                "phase_index": 2,
                "name": "concurrent_attacks",
                "scenario_type": self.name,
                "start_s": self.attack_start_s,
                "duration_s": self.attack_duration_s,
                "end_s": self.attack_start_s + self.attack_duration_s,
                "description": "Concurrent FDI (+5.0m) + DoS (33% drop) + Replay (5.0s buffer) across 2 circuit laps",
                "is_attack": True,
            },
            {
                "phase_index": 3,
                "name": "return_and_land",
                "scenario_type": "benign",
                "start_s": self.attack_start_s + self.attack_duration_s,
                "duration_s": 30.0,
                "end_s": self.attack_start_s + self.attack_duration_s + 30.0,
                "description": "Benign recovery: return-to-home and landing",
                "is_attack": False,
            },
        ]


class ScenarioPolicy:
    """Deterministic attack schedule and packet transformation parameters."""

    def __new__(cls, name, *args, **kwargs):
        if name == "sequence":
            inst = object.__new__(SequenceScenarioPolicy)
            inst.__init__(name, *args, **kwargs)
            return inst
        if name in ("combined", "concurrent"):
            inst = object.__new__(CombinedScenarioPolicy)
            inst.__init__(name, *args, **kwargs)
            return inst
        return super().__new__(cls)

    def __init__(self, name, duration_s, attack_start_s, attack_duration_s, config, fdi_subtype=None, replay_delay_s=None, replay_repetitions=None, fdi_offset_m=None, dos_drop_every=None):
        self.name = validate_scenario(name)
        self.duration_s = float(duration_s)
        self.attack_start_s = float(attack_start_s)
        self.attack_duration_s = float(attack_duration_s)
        self.config = config
        self.fdi_subtype = fdi_subtype or config["scenarios"]["fdi"]["subtype"]
        self.replay_delay_s = float(replay_delay_s if replay_delay_s is not None else config["scenarios"]["replay"]["delay_s"])
        self.replay_repetitions = int(replay_repetitions if replay_repetitions is not None else config["scenarios"]["replay"]["repetitions"])
        self.fdi_offset = float(fdi_offset_m if fdi_offset_m is not None else config["scenarios"]["fdi"]["altitude_offset_m"])
        self.dos_drop_every = int(dos_drop_every if dos_drop_every is not None else config["scenarios"]["dos"]["drop_every_nth_global_position"])
        if self.duration_s <= 0 or self.attack_start_s < 0 or self.attack_duration_s < 0:
            raise ValueError("Duration must be positive and attack schedule times cannot be negative")
        if self.name != "benign" and self.attack_start_s + self.attack_duration_s > self.duration_s:
            raise ValueError("The scheduled attack interval must fit inside the mission duration")
        if self.name == "FDI" and self.fdi_subtype not in ("step", "drift", "intermittent", "coordinated"):
            raise ValueError("FDI subtype must be step, drift, intermittent, or coordinated")
        if self.replay_delay_s <= 0 or self.replay_repetitions < 1:
            raise ValueError("Replay delay must be positive and repetitions must be at least one")

    def active_phase(self, mission_elapsed_s):
        return None

    def active(self, mission_elapsed_s):
        return self.name != "benign" and self.attack_start_s <= mission_elapsed_s < self.attack_start_s + self.attack_duration_s

    def active_scenario(self, mission_elapsed_s):
        return self.name if self.active(mission_elapsed_s) else "benign"

    def target_label(self, mission_elapsed_s):
        return self.name if self.active(mission_elapsed_s) else "Benign"

    def phases_summary(self):
        return [{
            "phase_index": 1,
            "name": self.name,
            "scenario_type": self.name,
            "start_s": self.attack_start_s if self.name != "benign" else 0.0,
            "duration_s": self.attack_duration_s if self.name != "benign" else self.duration_s,
            "end_s": (self.attack_start_s + self.attack_duration_s) if self.name != "benign" else self.duration_s,
            "is_attack": self.name != "benign",
        }]

    def fdi_offset_m(self, mission_elapsed_s):
        attack_elapsed = max(0.0, mission_elapsed_s - self.attack_start_s)
        settings = self.config["scenarios"]["fdi"]
        if self.fdi_subtype in ("step", "coordinated"):
            return float(self.fdi_offset)
        if self.fdi_subtype == "drift":
            return float(settings["drift_m_per_s"]) * attack_elapsed
        period = float(settings["intermittent_period_s"])
        return float(self.fdi_offset) if int(attack_elapsed / period) % 2 == 0 else 0.0


class ScenarioRelay(threading.Thread):
    """Receive MAVProxy UDP, apply an optional stream effect, and forward locally."""

    def __init__(self, run_id, policy: ScenarioPolicy, config, mavlink_module, events, error_list):
        super().__init__(name=f"acpdad-sitl-relay-{run_id}", daemon=True)
        self.run_id = run_id
        self.policy = policy
        self.config = config
        self.mavlink = mavlink_module
        self.events = events
        self.errors = error_list
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.flight_start_ns = None
        self.recent = deque()
        self.position_counter = 0
        self.datagram_counter = 0
        self.dropped = 0
        self.mutated = 0
        self.replayed = 0
        self.forwarded = 0
        self.bound_socket = None
        self.current_phase_name = ""
        self.on_phase_change = None

    def set_flight_start(self, monotonic_ns=None):
        self.flight_start_ns = int(monotonic_ns if monotonic_ns is not None else time.monotonic_ns())

    def _elapsed(self, arrival_ns):
        return None if self.flight_start_ns is None else max(0.0, (arrival_ns - self.flight_start_ns) / 1e9)

    def get_live_state(self) -> dict:
        now_ns = time.monotonic_ns()
        elapsed = self._elapsed(now_ns)
        phase = getattr(self.policy, "active_phase", lambda e: None)(elapsed)
        scenario = self.policy.active_scenario(elapsed) if hasattr(self.policy, "active_scenario") else ("benign" if not self.policy.active(elapsed or 0) else self.policy.name)
        active = self.policy.active(elapsed) if hasattr(self.policy, "active") else (scenario != "benign")
        injections_stopped = getattr(self.policy, "injections_stopped", False)

        if injections_stopped:
            state_label = "BENIGN_FORWARDING (INJECTIONS STOPPED)"
            active_injectors = []
        elif scenario in ("combined", "concurrent") and active:
            state_label = "CONCURRENT_ATTACKS_ACTIVE"
            active_injectors = ["FDI", "DoS", "Replay"]
        elif scenario != "benign" and active:
            state_label = f"INJECTING_{scenario.upper()}"
            active_injectors = [scenario]
        else:
            state_label = "BENIGN_FORWARDING"
            active_injectors = []

        return {
            "mission_elapsed_s": elapsed,
            "active_scenario": scenario,
            "injector_state": state_label,
            "active_injectors": active_injectors,
            "fdi_active": "FDI" in active_injectors,
            "dos_active": "DoS" in active_injectors,
            "replay_active": "Replay" in active_injectors,
            "injections_stopped": injections_stopped,
            "phase_name": phase.name if phase else ("concurrent_attacks" if active else "benign"),
            "phase_index": phase.phase_index if phase else (2 if active else 1),
            "phase_duration_s": phase.duration_s if phase else self.policy.duration_s,
            "forwarded": self.forwarded,
            "dropped": self.dropped,
            "mutated": self.mutated,
            "replayed": self.replayed,
        }

    def _rewrite_fdi(self, messages, elapsed_s):
        frames = []
        changed = False
        offset_m = self.policy.fdi_offset_m(elapsed_s)
        for message in messages:
            if _message_type(message) != "GLOBAL_POSITION_INT":
                frame = _message_frame(message)
                frames.append(frame)
                continue
            try:
                mav = self.mavlink.MAVLink(None, srcSystem=int(message.get_srcSystem()), srcComponent=int(message.get_srcComponent()))
                mav.seq = int(message.get_seq())
                altitude_offset_mm = int(round(offset_m * 1000.0))
                vz = int(message.vz)
                if self.policy.fdi_subtype == "coordinated":
                    vz -= int(round(float(self.config["scenarios"]["fdi"]["drift_m_per_s"]) * 100.0))
                updated = self.mavlink.MAVLink_global_position_int_message(
                    int(message.time_boot_ms), int(message.lat), int(message.lon),
                    int(message.alt) + altitude_offset_mm,
                    int(message.relative_alt) + altitude_offset_mm,
                    int(message.vx), int(message.vy), vz, int(message.hdg),
                )
                frame = bytes(updated.pack(mav))
                self.mutated += 1
                changed = True
            except Exception as exc:
                self.errors.append(f"FDI encode failure for seq={getattr(message, 'seq', '?')}: {type(exc).__name__}: {exc}")
                frame = _message_frame(message)
            frames.append(frame)
        return b"".join(frames), changed

    def _remember(self, elapsed_s, data, has_position):
        max_age = float(self.config["scenarios"]["replay"]["buffer_s"])
        self.recent.append((elapsed_s, bytes(data), has_position))
        while self.recent and (elapsed_s - self.recent[0][0] > max_age or len(self.recent) > 20000):
            self.recent.popleft()

    def process(self, data: bytes, arrival_ns: int):
        self.datagram_counter += 1
        try:
            messages = _parse_datagram(data, self.mavlink)
        except Exception as exc:
            messages = []
            self.errors.append(f"MAVLink parse failure: {type(exc).__name__}: {exc}")
        source_ms, sequence, message_type = _source_fields(messages)
        has_position = message_type == "GLOBAL_POSITION_INT"
        elapsed = self._elapsed(arrival_ns)
        active_scenario = self.policy.active_scenario(elapsed) if hasattr(self.policy, "active_scenario") else (self.policy.name if self.policy.active(elapsed or 0) else "benign")
        active = active_scenario != "benign"
        phase = getattr(self.policy, "active_phase", lambda e: None)(elapsed)
        phase_name = phase.name if phase is not None else ("attack" if active else "benign")
        phase_idx = phase.phase_index if phase is not None else 1

        if elapsed is not None and phase_name != self.current_phase_name:
            prev_phase = self.current_phase_name
            self.current_phase_name = phase_name
            self.position_counter = 0
            transition_ev = {
                "run_id": self.run_id,
                "event_type": "phase_transition",
                "from_phase": prev_phase,
                "to_phase": phase_name,
                "phase_index": phase_idx,
                "scenario": active_scenario,
                "mission_elapsed_s": elapsed,
                "arrival_time_ns": arrival_ns,
            }
            self.events.append(transition_ev)
            if callable(self.on_phase_change):
                try:
                    self.on_phase_change(transition_ev)
                except Exception:
                    pass

        replay_source = None
        output = [bytes(data)]
        effect = "forward"
        if elapsed is not None:
            self._remember(elapsed, data, has_position)
        if active and active_scenario == "DoS" and has_position:
            self.position_counter += 1
            every = int(getattr(phase, "dos_drop_every", self.policy.dos_drop_every))
            if every > 0 and self.position_counter % every == 0:
                output = []
                effect = "drop_detector_branch"
                self.dropped += 1
        elif active and active_scenario == "Replay" and has_position:
            replay_delay = float(getattr(phase, "replay_delay_s", self.policy.replay_delay_s))
            target_elapsed = elapsed - replay_delay
            candidates = [item for item in self.recent if item[2] and item[0] <= target_elapsed]
            if candidates:
                replay_elapsed, replay_data, _ = candidates[-1]
                replay_source = replay_elapsed
                reps = int(getattr(phase, "replay_repetitions", self.policy.replay_repetitions))
                output = [bytes(replay_data)] * reps
                effect = "replay_earlier_mavlink_datagram"
                self.replayed += len(output)
        elif active and active_scenario == "FDI" and has_position:
            rewritten, changed = self._rewrite_fdi(messages, elapsed)
            if changed:
                output = [rewritten]
                effect = "modify_global_position_altitude"
        elif active and active_scenario in ("combined", "concurrent") and has_position:
            self.position_counter += 1
            every = int(getattr(self.policy, "dos_drop_every", 3))
            # 1. DoS drop test: drop every Nth packet (default 33% bounded packet loss)
            if every > 0 and self.position_counter % every == 0:
                output = []
                effect = "drop_detector_branch"
                self.dropped += 1
            else:
                # 2. Interleave Replay (historical packet) and FDI (+5.0m step) across the remaining stream
                replay_delay = float(getattr(self.policy, "replay_delay_s", 5.0))
                target_elapsed = elapsed - replay_delay
                candidates = [item for item in self.recent if item[2] and item[0] <= target_elapsed]
                if self.position_counter % 2 == 1 and candidates:
                    replay_elapsed, replay_data, _ = candidates[-1]
                    replay_source = replay_elapsed
                    reps = int(getattr(self.policy, "replay_repetitions", 1))
                    output = [bytes(replay_data)] * reps
                    effect = "replay_earlier_mavlink_datagram"
                    self.replayed += len(output)
                else:
                    rewritten, changed = self._rewrite_fdi(messages, elapsed)
                    if changed:
                        output = [rewritten]
                        effect = "modify_global_position_altitude"
        event = {
            "run_id": self.run_id,
            "event_index": self.datagram_counter - 1,
            "input_arrival_time_ns": int(arrival_ns),
            "mission_elapsed_s": elapsed,
            "scheduled_scenario": active_scenario,
            "phase_name": phase_name,
            "phase_index": phase_idx,
            "attack_active": bool(active),
            "message_type": message_type,
            "source_time_ms": source_ms,
            "sequence": sequence,
            "input_sha256": _sha(data),
            "effect": effect,
            "replay_source_elapsed_s": replay_source,
            "outputs": [{"output_index": i, "sha256": _sha(payload), "length": len(payload)} for i, payload in enumerate(output)],
        }
        self.events.append(event)
        if effect == "forward":
            self.forwarded += len(output)
        return output

    def run(self):
        endpoint = self.config["endpoints"]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((endpoint["relay_bind_host"], int(endpoint["relay_bind_port"])))
        sock.settimeout(0.25)
        self.bound_socket = sock
        self.ready.set()
        output_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        target = (endpoint["detector_bind_host"], int(endpoint["detector_bind_port"]))
        try:
            while not self.stop_event.is_set():
                try:
                    data, address = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise
                if not address or not str(address[0]).startswith("127."):
                    self.errors.append(f"Rejected non-loopback MAVLink relay datagram from {address!r}")
                    continue
                arrival = time.monotonic_ns()
                for payload in self.process(data, arrival):
                    output_sock.sendto(payload, target)
        except Exception as exc:
            self.errors.append(f"Relay stopped: {type(exc).__name__}: {exc}")
        finally:
            output_sock.close()
            sock.close()

    def stop(self):
        self.stop_event.set()
        if self.bound_socket is not None:
            try:
                self.bound_socket.close()
            except OSError:
                pass


def ground_truth_row(run_id, scenario, mission_elapsed_s, policy):
    active = policy.active(float(mission_elapsed_s))
    if hasattr(policy, "target_label"):
        target_label = policy.target_label(float(mission_elapsed_s))
    elif scenario in ("combined", "concurrent"):
        target_label = "Combined_Attack" if active else "Benign"
    else:
        target_label = scenario if active else "Benign"
    return {
        "run_id": run_id,
        "mission_elapsed_s": float(mission_elapsed_s),
        "Target_Label": target_label,
        "attack_scheduled": bool(active),
        "label_source": "scenario harness schedule; joined after inference",
    }
