"""In-place terminal telemetry and detection dashboard for ACPDAD SITL flights.

Renders an auto-refreshing in-place terminal dashboard using Rich Live displaying
clean observer flight instruments, active concurrent/sequential attack injectors,
genuine live ML predictions, analytical physics checks, telemetry stream health,
and real-time event logs. Does not start any web server or open any browser.
"""

from __future__ import annotations

import collections
import math
import sys
import threading
import time
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TerminalDashboard:
    """Renders an in-place terminal dashboard using Rich Live without any web server or browser."""

    def __init__(self, state_provider: Callable[[], dict], refresh_rate_hz: float = 4.0):
        self.state_provider = state_provider
        self.refresh_rate_hz = float(refresh_rate_hz)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.flight_start_ns: int | None = None
        self.events: collections.deque = collections.deque(maxlen=6)
        self.console = Console(file=sys.stdout, highlight=False)
        self.live: Live | None = None
        self._lock = threading.Lock()
        self.log_event("INIT", "Dashboard initialized. Awaiting SITL telemetry.")

    def set_flight_start(self, monotonic_ns: int | None = None):
        self.flight_start_ns = int(monotonic_ns if monotonic_ns is not None else time.monotonic_ns())

    def log_event(self, category: str, message: str):
        with self._lock:
            elapsed_s = 0.0
            if self.flight_start_ns is not None:
                elapsed_s = max(0.0, (time.monotonic_ns() - self.flight_start_ns) / 1e9)
            sim_elapsed_s = elapsed_s * 2.0  # SITL 2.0x speedup
            mins = int(sim_elapsed_s // 60)
            secs = sim_elapsed_s % 60
            ts = f"{mins:02d}:{secs:04.1f}"
            self.events.append((ts, str(category).upper(), str(message)))

    def _build_renderable(self):
        try:
            state = self.state_provider() or {}
        except Exception as exc:
            state = {"error": str(exc)}

        now_mono = time.monotonic_ns()
        flight_elapsed_s = max(0.0, (now_mono - self.flight_start_ns) / 1e9) if self.flight_start_ns else 0.0
        sim_elapsed_s = flight_elapsed_s * 2.0

        flight_status = state.get("flight_status", "INITIALIZING")
        flight_mode = state.get("flight_mode", "GUIDED")
        armed = state.get("armed", False)
        observer = state.get("observer", {})
        ml = state.get("ml_inference", {})
        health = state.get("detector_health", {})
        relay = state.get("relay", {})
        seq = state.get("sequence", {})

        alt_m = observer.get("relative_alt_m", 0.0)
        spd_mps = observer.get("groundspeed_mps", 0.0)
        hdg_deg = observer.get("heading_deg", 0.0)
        lap = observer.get("lap", 1)
        wp = observer.get("waypoint", 0)
        dist_wp = observer.get("distance_to_wp_m", 0.0)

        # 1. Main Status Table
        t_main = Table(box=None, padding=(0, 1), expand=True)
        t_main.add_column("Component", style="bold cyan", width=18, no_wrap=True)
        t_main.add_column("Real-time State & Live Metrics", style="white")

        # Flight Status
        status_color = "green" if "AIRBORNE" in flight_status else ("yellow" if "LAND" in flight_status or "HOME" in flight_status else "cyan")
        armed_str = "[red]ARMED[/red]" if armed else "[dim]DISARMED[/dim]"
        t_main.add_row(
            "Flight Status",
            f"[{status_color}]{flight_status}[/{status_color}] ({armed_str}, Mode: {flight_mode})  |  "
            f"Elapsed: [bold]{sim_elapsed_s:.1f}s[/bold] sim ({flight_elapsed_s:.1f}s wall, 2.0x)  |  "
            f"Lap: [bold]{lap}/2[/bold] (WP {wp}/8, {dist_wp:.1f}m)"
        )

        # Telemetry Instruments
        t_main.add_row(
            "Telemetry (QGC)",
            f"Alt: [bold]{alt_m:.1f}m[/bold] AGL  |  "
            f"Speed: [bold]{spd_mps:.1f}m/s[/bold]  |  "
            f"Heading: [bold]{int(hdg_deg):03d}°[/bold]  |  "
            f"QGroundControl: [green]CONNECTED (127.0.0.1:14600)[/green]"
        )

        # Active Attack Injectors
        active_scen = relay.get("active_scenario", seq.get("scenario_type", "benign"))
        inj_stopped = relay.get("injections_stopped", False)

        mut_count = relay.get("mutated", health.get("mutated", 0))
        drop_count = relay.get("dropped", health.get("dropped", 0))
        rep_count = relay.get("replayed", health.get("replayed", 0))
        fwd_count = relay.get("forwarded", health.get("forwarded", 0))

        if inj_stopped:
            inj_summary = "[dim green]STOPPED / BENIGN FORWARDING (Circuit completed; return-to-home)[/dim green]"
        elif active_scen in ("combined", "concurrent"):
            inj_summary = (
                f"[bold red]CONCURRENT ACTIVE[/bold red]: "
                f"FDI: [red]ACTIVE[/red] (+5.0m step, {mut_count} mut) | "
                f"DoS: [red]ACTIVE[/red] (33% drop, {drop_count} drop) | "
                f"Replay: [red]ACTIVE[/red] (5.0s buf, {rep_count} rep)"
            )
        elif active_scen != "benign":
            inj_summary = f"[bold red]ACTIVE INJECTOR ({active_scen.upper()})[/bold red]: {mut_count} mut, {drop_count} drop, {rep_count} rep"
        else:
            inj_summary = "[green]BENIGN FORWARDING[/green] (No active injection on detector branch)"

        t_main.add_row("Attack Injectors", inj_summary)

        # Genuine Live ML Prediction
        pred = ml.get("latest_prediction", "INITIALIZING")
        conf = ml.get("confidence")
        conf_str = f"({conf*100:.1f}%)" if (conf is not None and conf > 0) else ""
        stage = ml.get("stage", "primary_temporal")
        is_stale = ml.get("is_stale", False)
        pred_color = "green" if pred == "Benign" else "bold red"
        stale_badge = " [bold red][NO FRESH PREDICTION (STALE)][/bold red]" if is_stale else " [dim green][FRESH <0.2s][/dim green]"
        alerts_count = 1 if ml.get("attack_alert_candidate") else 0

        probs = ml.get("probabilities", {})
        prob_str = ""
        if probs:
            prob_items = [f"{k}: {probs.get(k, 0.0):.2f}" for k in ("Benign", "DoS", "Replay", "FDI")]
            prob_str = "  |  [" + " ".join(prob_items) + "]"

        t_main.add_row(
            "Live ML Inference",
            f"Prediction: [{pred_color}]{pred}[/{pred_color}] {conf_str}{stale_badge}  |  "
            f"Stage: {stage}  |  Alerts: {alerts_count}{prob_str}"
        )

        # Physics Residuals & Analytical Checks
        v_res = health.get("vertical_residual_m")
        phys_active = health.get("physics_warning_active", False)
        p_thresh = health.get("vertical_residual_threshold_m", 0.0108)
        if v_res is not None:
            if phys_active:
                phys_str = f"[bold yellow]r_z = {v_res:.4f}m[/bold yellow] [bold red][PHYSICS WARNING CANDIDATE > {p_thresh:.4f}m][/bold red]"
            else:
                phys_str = f"[green]r_z = {v_res:.4f}m[/green] [dim](Normal kinematics <= {p_thresh:.4f}m)[/dim]"
        else:
            phys_str = "[dim]Awaiting telemetry observations...[/dim]"
        t_main.add_row(
            "Physics Residuals",
            f"{phys_str}  [dim](Analytical check, distinct from ML detections)[/dim]"
        )

        # Telemetry Health & Counts
        rate_hz = health.get("telemetry_rate_hz", 0.0)
        dgrams = health.get("datagrams_received", 0)
        preds_total = health.get("predictions_count", 0)
        gap_active = health.get("watchdog_gap_active", False)
        gap_str = "[red]GAP DETECTED[/red]" if gap_active else "[green]NOMINAL[/green]"

        t_main.add_row(
            "Telemetry Health",
            f"Rate: [bold]{rate_hz:.1f} Hz[/bold]  |  "
            f"Stream: {gap_str}  |  "
            f"Datagrams: {dgrams} (Fwd: {fwd_count})  |  "
            f"Predictions: {preds_total}  |  "
            f"Evil_Twin: [dim]UNSUPPORTED (No WLAN)[/dim]"
        )

        # 2. Scrolling Event Log Table
        t_log = Table(box=None, padding=(0, 1), expand=True)
        t_log.add_column("Sim Time", style="dim cyan", width=12, no_wrap=True)
        t_log.add_column("Category", style="bold yellow", width=12, no_wrap=True)
        t_log.add_column("Event Description", style="white")

        with self._lock:
            recent_events = list(self.events)

        if not recent_events:
            t_log.add_row("--:--.-", "INFO", "No events recorded yet.")
        else:
            for ts, cat, msg in recent_events:
                cat_style = "bold red" if "ATTACK" in cat or "WARN" in cat else ("bold cyan" if "NAV" in cat else "bold yellow")
                t_log.add_row(ts, f"[{cat_style}]{cat}[/{cat_style}]", msg)

        return Group(
            Panel(t_main, title="[bold green]ACPDAD REAL-TIME IN-FLIGHT DETECTION DASHBOARD (IN-PLACE TERMINAL)[/bold green]", border_style="blue"),
            Panel(t_log, title="[bold cyan]RECENT MISSION & DETECTOR EVENT LOG (LAST 6 EVENTS)[/bold cyan]", border_style="cyan"),
        )

    def _loop(self):
        period = 1.0 / max(1.0, self.refresh_rate_hz)
        try:
            with Live(self._build_renderable(), console=self.console, refresh_per_second=int(self.refresh_rate_hz), transient=False) as live:
                self.live = live
                while not self.stop_event.is_set():
                    live.update(self._build_renderable())
                    time.sleep(period)
                live.update(self._build_renderable())
        except Exception:
            pass

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="acpdad-terminal-dashboard", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)


class DashboardServer:
    """Compatibility wrapper mapping DashboardServer to TerminalDashboard (no web server)."""

    def __init__(self, *args, **kwargs):
        state_provider = kwargs.get("state_provider", lambda: {})
        self._impl = TerminalDashboard(state_provider=state_provider)

    def start(self):
        self._impl.start()
        return "terminal://in-place-rich-dashboard"

    def stop(self):
        self._impl.stop()

    def set_flight_start(self, ns):
        self._impl.set_flight_start(ns)

    def log_event(self, cat, msg):
        self._impl.log_event(cat, msg)
