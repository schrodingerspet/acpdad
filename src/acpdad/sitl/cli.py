"""Command-line interface for the isolated ArduPilot SITL/MAVLink workflow."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import SITL_SCHEMA_VERSION
from .detector import infer_saved_capture
from .runner import (
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    PROJECT_ROOT,
    capture_session,
    check_prerequisites,
    controller_mission,
    run_session,
    start_session,
    stop_session,
)
from .verification import run_checks


def _parser():
    parser = argparse.ArgumentParser(
        prog="run_project.py sitl",
        description="ArduPilot SITL + localhost MAVLink workflow (separate from ACPDAD synthetic data).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check pinned SITL/MAVLink prerequisites and local paths")
    doctor.add_argument("--ardupilot-home", type=Path)
    doctor.add_argument("--json", action="store_true")

    verify = sub.add_parser("selfcheck", help="Check feature mapping, state, fallback, checkpoint inference and scenario boundaries")
    verify.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    verify.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "verification")

    start = sub.add_parser("start", help="Start ArduPilot SITL and the MAVProxy loopback bridge")
    start.add_argument("--ardupilot-home", type=Path)
    start.add_argument("--session-dir", type=Path)
    start.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    start.add_argument("--headless", action="store_true", help="Run headlessly without requiring QGroundControl visualization")
    start.add_argument("--qgroundcontrol", "--qgc", dest="qgroundcontrol", action=argparse.BooleanOptionalAction, default=None, help="Forward live telemetry to QGroundControl (required by default unless --headless)")
    start.add_argument("--qgroundcontrol-port", "--qgc-port", dest="qgroundcontrol_port", type=int, default=None, help="Explicit UDP port for QGroundControl (defaults to 14600 + instance)")

    stop = sub.add_parser("stop", help="Stop an ACPDAD SITL session cleanly")
    stop.add_argument("--session-dir", type=Path, required=True)

    control = sub.add_parser("controller", help="Connect through the MAVProxy localhost UDP route and fly a benign takeoff/land mission")
    control.add_argument("--session-dir", type=Path, required=True)
    control.add_argument("--duration", type=float, default=30.0)
    control.add_argument("--altitude", type=float)
    control.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    capture = sub.add_parser("capture", help="Capture the detector-facing MAVLink relay stream while a session is running")
    capture.add_argument("--session-dir", type=Path, required=True)
    capture.add_argument("--scenario", choices=("benign", "FDI", "DoS", "Replay", "Evil_Twin", "sequence", "combined", "concurrent"), default="benign")
    capture.add_argument("--duration", type=float, default=30.0)
    capture.add_argument("--attack-start", type=float)
    capture.add_argument("--attack-duration", type=float)
    capture.add_argument("--fdi-subtype", choices=("step", "drift", "intermittent", "coordinated"))
    capture.add_argument("--replay-delay", type=float)
    capture.add_argument("--replay-repetitions", type=int)
    capture.add_argument("--output", type=Path)
    capture.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    capture.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    run = sub.add_parser("run", help="Run a complete SITL -> controller + MAVProxy -> detector mission and save artifacts")
    run.add_argument("--scenario", choices=("benign", "FDI", "DoS", "Replay", "Evil_Twin", "sequence", "combined", "concurrent"), default="benign")
    run.add_argument("--duration", type=float, default=30.0, help="Seconds airborne before the controller lands")
    run.add_argument("--attack-start", type=float)
    run.add_argument("--attack-duration", type=float)
    run.add_argument("--fdi-subtype", choices=("step", "drift", "intermittent", "coordinated"))
    run.add_argument("--replay-delay", type=float)
    run.add_argument("--replay-repetitions", type=int)
    run.add_argument("--altitude", type=float, default=15.0, help="Target hover/route altitude in meters (default 15.0)")
    run.add_argument("--speed", dest="cruise_speed_mps", type=float, default=16.0, help="Circuit cruise speed in m/s (default 16.0)")
    run.add_argument("--north-m", type=float, default=180.0, help="Half-length north/south span in meters (default 180.0)")
    run.add_argument("--east-m", type=float, default=120.0, help="Half-length east/west span in meters (default 120.0)")
    run.add_argument("--fdi-offset", type=float, help="FDI altitude offset in meters")
    run.add_argument("--dos-drop-every", type=int, help="Drop every Nth GLOBAL_POSITION_INT during DoS")
    run.add_argument("--headless", action="store_true", help="Run headlessly without requiring QGroundControl visualization")
    run.add_argument("--qgroundcontrol", "--qgc", dest="qgroundcontrol", action=argparse.BooleanOptionalAction, default=None, help="Forward live telemetry to QGroundControl (required by default unless --headless)")
    run.add_argument("--qgroundcontrol-port", "--qgc-port", dest="qgroundcontrol_port", type=int, default=None, help="Explicit UDP port for QGroundControl (defaults to 14600 + instance)")
    run.add_argument("--dashboard", dest="dashboard", action=argparse.BooleanOptionalAction, default=True, help="Run live HTTP telemetry dashboard on localhost (default True)")
    run.add_argument("--dashboard-port", dest="dashboard_port", type=int, default=8088, help="Port for the live HTTP dashboard (default 8088)")
    run.add_argument("--route", dest="route", action=argparse.BooleanOptionalAction, default=None, help="Fly repeatable multi-lap circuit route (default True)")
    run.add_argument("--laps", type=int, default=2, help="Number of route laps to fly (default 2)")
    run.add_argument("--radius", type=float, default=None, help="Legacy perimeter route radius in meters")
    run.add_argument("--pause", dest="pause_at_wp_s", type=float, default=0.5, help="Pause duration at each waypoint in seconds (default 0.5)")
    run.add_argument("--post-hold", dest="post_landing_hold_s", type=float, default=15.0, help="Post-landing hold duration in seconds to keep telemetry open (default 15.0)")
    run.add_argument("--run-id")
    run.add_argument("--ardupilot-home", type=Path)
    run.add_argument("--output-root", type=Path)
    run.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    infer = sub.add_parser("infer", help="Re-run label-free inference over one saved SITL capture")
    infer.add_argument("--run-dir", type=Path, required=True, help="Capture directory containing features.csv")
    infer.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    infer.add_argument("--output", type=Path)

    schema = sub.add_parser("schema", help="Print the source-specific SITL mapping schema version")
    schema.add_argument("--output", type=Path)

    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = check_prerequisites(args.ardupilot_home)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"ACPDAD SITL readiness: {'READY' if result['ready'] else 'NOT READY'}")
                for name, value in result.items():
                    if name != "ready":
                        print(f"- {name}: {'PASS' if value.get('ok') else 'MISSING/UNVERIFIED'} {value}")
                if not result["ready"]:
                    print("Install instructions: docs/SITL_MAVLINK.md")
            return 0 if result["ready"] else 1
        if args.command == "selfcheck":
            result = run_checks(args.model, args.output)
            print(json.dumps(result, indent=2))
            return 0 if result["failed"] == 0 else 1
        if args.command == "start":
            session_dir, result = start_session(
                args.ardupilot_home, args.session_dir, args.config,
                headless=args.headless,
                qgroundcontrol=args.qgroundcontrol,
                qgroundcontrol_port=args.qgroundcontrol_port,
            )
            print(json.dumps({"status": result["status"], "session_dir": str(session_dir), "endpoints": result["endpoints"], "provenance": result["provenance"], "qgroundcontrol": result.get("qgroundcontrol")}, indent=2))
            print("Run controller, capture or stop from another terminal. Processes are recorded in session.json.")
            return 0
        if args.command == "stop":
            result = stop_session(args.session_dir)
            print(json.dumps({"status": result["status"], "session_id": result["session_id"], "stopped_utc": result["stopped_utc"]}, indent=2))
            return 0
        if args.command == "controller":
            print(json.dumps(controller_mission(args.session_dir, args.duration, args.altitude, args.config), indent=2))
            return 0
        if args.command == "capture":
            print(json.dumps(capture_session(
                args.session_dir, args.scenario, args.duration, args.output,
                args.attack_start, args.attack_duration, args.fdi_subtype,
                args.replay_delay, args.replay_repetitions, args.model, args.config,
            ), indent=2))
            return 0
        if args.command == "run":
            print(json.dumps(run_session(
                args.scenario, args.duration, args.attack_start, args.attack_duration,
                args.fdi_subtype, args.replay_delay, args.replay_repetitions,
                args.run_id, args.ardupilot_home, args.output_root, args.model, args.config,
                args.altitude, args.fdi_offset, args.dos_drop_every,
                headless=args.headless,
                qgroundcontrol=args.qgroundcontrol,
                qgroundcontrol_port=args.qgroundcontrol_port,
                route=args.route,
                laps=args.laps,
                north_m=args.north_m,
                east_m=args.east_m,
                cruise_speed_mps=args.cruise_speed_mps,
                pause_at_wp_s=args.pause_at_wp_s,
                post_landing_hold_s=args.post_landing_hold_s,
                radius_m=args.radius,
                dashboard=args.dashboard,
                dashboard_port=args.dashboard_port,
            ), indent=2))
            return 0
        if args.command == "infer":
            output = args.output or (args.run_dir / "predictions_reinferred_label_free.csv")
            print(json.dumps(infer_saved_capture(args.run_dir, args.model, output), indent=2))
            return 0
        if args.command == "schema":
            from .adapter import sitl_schema
            content = json.dumps(sitl_schema(), indent=2) + "\n"
            if args.output:
                args.output.write_text(content)
                print(str(args.output))
            else:
                print(content)
            return 0
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"ACPDAD SITL error: {exc}", file=sys.stderr)
        return 2
    return 0
