"""Orchestrate live ArduPilot SITL + QGroundControl balanced multi-lap circuit flight and capture milestone screenshots."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acpdad.sitl.runner import run_session

RUN_ID = "sitl_qgc_route_demo_003"
TMP_DIR = Path("/tmp/sitl_demo_screenshots")
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)
TMP_DIR.mkdir(parents=True, exist_ok=True)


def capture_screenshot(tag: str):
    target = TMP_DIR / f"{tag}.png"
    env = dict(os.environ)
    env["WAYLAND_DISPLAY"] = os.environ.get("WAYLAND_DISPLAY", "wayland-1")
    try:
        subprocess.run(["grim", "-t", "png", str(target)], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[Screenshot] Captured {tag}: {target.name} ({target.stat().st_size} bytes)", file=sys.stderr)
    except Exception as exc:
        print(f"[Screenshot] Failed to capture {tag}: {exc}", file=sys.stderr)


def screenshot_worker(stop_event: threading.Event, start_time: float):
    schedule = [
        (18.0, "01_takeoff_climb_15m"),
        (35.0, "02_in_flight_lap1_eastern_leg"),
        (55.0, "03_in_flight_lap2_western_leg"),
    ]
    captured = set()
    while not stop_event.is_set():
        now = time.monotonic()
        elapsed = now - start_time
        for target_sec, tag in schedule:
            if tag not in captured and elapsed >= target_sec:
                capture_screenshot(tag)
                captured.add(tag)
        time.sleep(1.0)


def on_landed():
    print("[Flight] Touchdown & disarm confirmed! Entering post-landing hold period...", file=sys.stderr)
    time.sleep(2.0)
    capture_screenshot("04_post_landing_runway_full_circuit_trail")
    time.sleep(4.0)
    capture_screenshot("05_post_landing_telemetry_active")


def main():
    print("=" * 70, file=sys.stderr)
    print("ACPDAD ArduPilot SITL + QGroundControl Flight Demonstration", file=sys.stderr)
    print(f"Run ID: {RUN_ID}", file=sys.stderr)
    print("Circuit: 8 waypoints, 180m North x 120m East (~1.0 km/lap, 2 laps = ~2.0 km)", file=sys.stderr)
    print("Cruise Speed: 16.0 m/s | Altitude: 15.0m AGL", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    stop_event = threading.Event()
    start_time = time.monotonic()
    worker = threading.Thread(target=screenshot_worker, args=(stop_event, start_time), daemon=True)
    worker.start()

    run_result = None
    try:
        run_result = run_session(
            scenario="benign",
            run_id=RUN_ID,
            laps=2,
            north_m=180.0,
            east_m=120.0,
            altitude_m=15.0,
            cruise_speed_mps=16.0,
            acceptance_radius_m=15.0,
            pause_at_wp_s=0.5,
            post_landing_hold_s=15.0,
            on_landed=on_landed,
        )
        print("\n[Runner] SITL mission completed successfully!", file=sys.stderr)
    finally:
        stop_event.set()
        worker.join(timeout=3.0)

    # Copy captured screenshots into the run output directory
    run_dir = PROJECT_ROOT / "results" / "sitl" / "runs" / RUN_ID
    dest_screenshots = run_dir / "screenshots"
    dest_screenshots.mkdir(parents=True, exist_ok=True)
    for p in sorted(TMP_DIR.glob("*.png")):
        shutil.copy2(p, dest_screenshots / p.name)
        print(f"[Artifact] Saved screenshot to {dest_screenshots / p.name}", file=sys.stderr)

    print("\nFlight Summary:", file=sys.stderr)
    if run_result:
        manifest_path = Path(run_result["run_manifest"])
        if manifest_path.is_file():
            import json
            manifest = json.loads(manifest_path.read_text())
            route = manifest.get("route", {})
            print(f"- Status: {manifest.get('status')}", file=sys.stderr)
            print(f"- Laps completed: {route.get('laps')}", file=sys.stderr)
            print(f"- Lap distance: {route.get('lap_distance_m')} m", file=sys.stderr)
            print(f"- Total circuit distance: {route.get('total_distance_m')} m", file=sys.stderr)
            print(f"- Cruise speed: {route.get('cruise_speed_mps')} m/s", file=sys.stderr)
            print(f"- Waypoints completed: {route.get('waypoints_completed')}", file=sys.stderr)
            print(f"- Output directory: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
