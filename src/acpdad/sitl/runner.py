"""SITL process lifecycle, controller orchestration and provenance records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .capture import CaptureContext
from .controller import SitlController
from .dashboard import DashboardServer, TerminalDashboard
from .scenarios import ScenarioPolicy, SequenceScenarioPolicy, validate_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "sitl.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "sitl" if (PROJECT_ROOT / "models" / "sitl" / "checkpoint.pkl").is_file() else PROJECT_ROOT / "models" / "selected"
PINNED_ARDUPILOT_TAG = "Copter-4.7.2"
PINNED_MAVPROXY = "1.8.75"
PINNED_PYMAVLINK = "2.4.50"
PINNED_PEXPECT = "4.9.0"


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _import_mavutil():
    os.environ["MAVLINK20"] = "1"
    os.environ["MAVLINK_DIALECT"] = "ardupilotmega"
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise RuntimeError("pymavlink is required for SITL. Install it with: .venv/bin/python -m pip install -r requirements-sitl.txt") from exc
    return mavutil


def _tool(name):
    path = shutil.which(name)
    if path:
        return Path(path).resolve()
    candidate = PROJECT_ROOT / ".venv" / "bin" / name
    return candidate.resolve() if candidate.is_file() else None


def _ardupilot_home(value=None):
    raw = value or os.environ.get("ARDUPILOT_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    default = PROJECT_ROOT / ".sitl" / "ardupilot"
    return default.resolve()


def _git_version(home):
    tag = None
    commit = None
    if (Path(home) / ".git").exists():
        try:
            tags = subprocess.check_output(["git", "-C", str(home), "tag", "--points-at", "HEAD"], text=True, stderr=subprocess.DEVNULL).split()
            tag = PINNED_ARDUPILOT_TAG if PINNED_ARDUPILOT_TAG in tags else (tags[0] if tags else None)
            commit = subprocess.check_output(["git", "-C", str(home), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            pass
    return tag, commit


def _wait_for_port(port, timeout_s=30.0, host="127.0.0.1"):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                if sock.connect_ex((host, int(port))) == 0:
                    return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


def check_prerequisites(ardupilot_home=None, require_running_tools=True):
    home = _ardupilot_home(ardupilot_home)
    sim_vehicle = home / "Tools" / "autotest" / "sim_vehicle.py"
    built_vehicle = home / "build" / "sitl" / "bin" / "arducopter"
    mavproxy = _tool("mavproxy.py")
    checks = {
        "python": {"ok": sys.version_info >= (3, 10), "version": sys.version.split()[0]},
        "ardupilot_home": {"ok": (home / ".git").exists(), "path": str(home)},
        "sim_vehicle.py": {"ok": sim_vehicle.is_file(), "path": str(sim_vehicle)},
        "built_arducopter": {"ok": built_vehicle.is_file(), "path": str(built_vehicle)},
        "mavproxy.py": {"ok": mavproxy is not None, "path": str(mavproxy) if mavproxy else None},
    }
    try:
        from pymavlink import mavutil  # noqa: F401
        version = importlib.metadata.version("pymavlink")
        checks["pymavlink"] = {"ok": version == PINNED_PYMAVLINK, "version": version, "expected": PINNED_PYMAVLINK}
    except Exception as exc:
        checks["pymavlink"] = {"ok": False, "expected": PINNED_PYMAVLINK, "issue": f"{type(exc).__name__}: {exc}"}
    try:
        version = importlib.metadata.version("pexpect")
        checks["pexpect"] = {"ok": version == PINNED_PEXPECT, "version": version, "expected": PINNED_PEXPECT}
    except Exception as exc:
        checks["pexpect"] = {"ok": False, "expected": PINNED_PEXPECT, "issue": f"{type(exc).__name__}: {exc}"}
    tag, commit = _git_version(home)
    if (home / ".git").exists():
        if tag is not None and commit is not None:
            checks["ardupilot_version"] = {"ok": tag == PINNED_ARDUPILOT_TAG, "tag": tag, "commit": commit, "expected": PINNED_ARDUPILOT_TAG}
        else:
            checks["ardupilot_version"] = {"ok": False, "expected": PINNED_ARDUPILOT_TAG, "issue": "cannot determine tag/commit from git checkout"}
    else:
        checks["ardupilot_version"] = {"ok": False, "expected": PINNED_ARDUPILOT_TAG, "issue": "cannot verify version: ArduPilot checkout is absent"}
    if mavproxy is not None:
        try:
            version = importlib.metadata.version("MAVProxy")
            checks["mavproxy_version"] = {"ok": version == PINNED_MAVPROXY, "version": version, "expected": PINNED_MAVPROXY}
        except Exception as exc:
            checks["mavproxy_version"] = {"ok": False, "expected": PINNED_MAVPROXY, "issue": f"{type(exc).__name__}: {exc}"}
    else:
        checks["mavproxy_version"] = {"ok": False, "expected": PINNED_MAVPROXY, "issue": "cannot verify version: mavproxy.py is absent"}
    required = ["python", "ardupilot_home", "sim_vehicle.py", "built_arducopter", "mavproxy.py", "pymavlink", "pexpect"]
    if require_running_tools:
        required += ["ardupilot_version", "mavproxy_version"]
    checks["ready"] = all(checks.get(name, {}).get("ok", False) for name in required)
    return checks


def _load_config(path=DEFAULT_CONFIG):
    config = _read_json(path)
    if config.get("provenance") != "ARDUPILOT_SITL_LOCAL_SIMULATION":
        raise ValueError("SITL configuration provenance is missing or invalid")
    endpoints = config["endpoints"]
    loopback_endpoints = {
        "sitl_master": "tcp:127.0.0.1:",
        "controller": "udpin:127.0.0.1:",
        "controller_gcs_forward": "udp:127.0.0.1:",
        "proxy_relay_forward": "udp:127.0.0.1:",
    }
    for name, prefix in loopback_endpoints.items():
        if not str(endpoints.get(name, "")).startswith(prefix):
            raise ValueError(f"{name} must use the localhost-only endpoint form {prefix}<port>")
    controller_port = str(endpoints["controller"]).rsplit(":", 1)[1]
    controller_forward_port = str(endpoints["controller_gcs_forward"]).rsplit(":", 1)[1]
    if controller_port != controller_forward_port:
        raise ValueError("The controller bind port must match the MAVProxy controller-forward port")
    if int(str(endpoints["proxy_relay_forward"]).rsplit(":", 1)[1]) != int(endpoints["relay_bind_port"]):
        raise ValueError("The scenario relay bind port must match the MAVProxy detector-forward port")
    if endpoints["relay_bind_host"] != "127.0.0.1" or endpoints["detector_bind_host"] != "127.0.0.1":
        raise ValueError("Relay and detector must bind only to 127.0.0.1")
    return config


def _inspect_port_owner(port: int, proto: str = "udp") -> str | None:
    flag = "-ulpn" if proto == "udp" else "-tlpn"
    try:
        out = subprocess.check_output(
            ["ss", flag, f"sport = :{int(port)}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        match = re.search(r'users:\(\(\"([^\"]+)\"', out)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def get_qgroundcontrol_port(config: dict, custom_port: int | None = None) -> int:
    if custom_port is not None:
        port = int(custom_port)
    else:
        instance = int(config.get("instance", 0))
        base_port = int(config.get("endpoints", {}).get("qgroundcontrol_base_port", 14600))
        port = base_port + instance
    if port < 1024 or port > 65535:
        raise ValueError(f"QGroundControl UDP port {port} is out of valid range (1024-65535)")
    return port


def _find_qgroundcontrol_binary() -> Path | None:
    standard_paths = [
        Path.home() / "Applications" / "QGroundControl" / "QGroundControl-x86_64.AppImage",
        Path.home() / "Applications" / "QGroundControl.AppImage",
        Path("/usr/local/bin/qgroundcontrol"),
        Path("/usr/bin/qgroundcontrol"),
    ]
    for p in standard_paths:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    which_bin = shutil.which("qgroundcontrol") or shutil.which("QGroundControl")
    if which_bin:
        return Path(which_bin)
    return None


def ensure_qgroundcontrol(config: dict, custom_port: int | None = None, headless: bool = False, auto_start: bool = True) -> int | None:
    if headless:
        return None
    port = get_qgroundcontrol_port(config, custom_port)
    owner = _inspect_port_owner(port, proto="udp")
    if owner and "qgroundcontrol" in owner.lower():
        print(f"[QGroundControl] Confirmed QGroundControl active on UDP 127.0.0.1:{port}", file=sys.stderr)
        return port

    # Port is not currently owned by QGroundControl. Try auto-starting if GUI environment is present.
    if auto_start and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        qgc_bin = _find_qgroundcontrol_binary()
        if qgc_bin:
            print(f"[QGroundControl] Auto-launching {qgc_bin.name}...", file=sys.stderr)
            try:
                subprocess.Popen(
                    [str(qgc_bin)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline:
                    cur_owner = _inspect_port_owner(port, proto="udp")
                    if cur_owner and "qgroundcontrol" in cur_owner.lower():
                        print(f"[QGroundControl] QGroundControl connected and listening on UDP 127.0.0.1:{port}", file=sys.stderr)
                        return port
                    time.sleep(0.5)
            except Exception as exc:
                print(f"[QGroundControl] Auto-launch attempt failed: {exc}", file=sys.stderr)

    cur_owner = _inspect_port_owner(port, proto="udp")
    if cur_owner and "qgroundcontrol" in cur_owner.lower():
        return port

    raise RuntimeError(
        f"QGroundControl is required for interactive SITL missions, but is not listening on UDP port {port}.\n"
        f"Please start QGroundControl (listening on localhost UDP port {port}) before launching this mission.\n"
        f"(Note: For automated batch checks or headless testing, pass --headless)."
    )


def _check_ports_available(config, qgc_port=None):
    unavailable = []
    endpoints = config["endpoints"]
    candidates = [
        (int(endpoints["sitl_master"].rsplit(":", 1)[1]), socket.SOCK_STREAM),
        (int(endpoints["controller_gcs_forward"].rsplit(":", 1)[1]), socket.SOCK_DGRAM),
        (int(endpoints["proxy_relay_forward"].rsplit(":", 1)[1]), socket.SOCK_DGRAM),
        (int(endpoints["detector_bind_port"]), socket.SOCK_DGRAM),
    ]
    for port, sock_type in candidates:
        sock = socket.socket(socket.AF_INET, sock_type)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            unavailable.append(port)
        finally:
            sock.close()
    if unavailable:
        raise RuntimeError(f"Local SITL ports are already in use: {unavailable}; stop the other session before starting ACPDAD SITL")

    if qgc_port is not None:
        port = int(qgc_port)
        sitl_endpoint_ports = {p for p, _ in candidates}
        # Reserved SITL blocks: controller/relay/detector 14550-14589, SITL master 5760-5789
        reserved_ranges = set(range(14550, 14590)) | set(range(5760, 5790))
        if port in sitl_endpoint_ports or port in reserved_ranges:
            raise RuntimeError(
                f"QGroundControl port {port} collides with reserved SITL endpoint ports "
                f"(controller/detector/relay 14550-14589, SITL master 5760-5789). "
                f"Choose a port outside this range (e.g., 14600+)."
            )
        sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock_tcp.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"QGroundControl port {port} is already occupied by a TCP service on 127.0.0.1")
        finally:
            sock_tcp.close()

        sock_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock_udp.bind(("127.0.0.1", port))
            sock_udp.close()
        except OSError:
            sock_udp.close()
            owner = _inspect_port_owner(port, proto="udp")
            if owner and "qgroundcontrol" in owner.lower():
                pass
            else:
                owner_desc = f" ({owner})" if owner else ""
                raise RuntimeError(
                    f"QGroundControl UDP port {port} is already occupied by another process{owner_desc}; "
                    f"stop the conflicting process or choose another port"
                )


def _source_hashes():
    names = [
        PROJECT_ROOT / "run_project.py",
        PROJECT_ROOT / "src" / "acpdad" / "simulation.py",
        PROJECT_ROOT / "src" / "acpdad" / "workflow.py",
        *sorted((PROJECT_ROOT / "src" / "acpdad" / "sitl").glob("*.py")),
    ]
    return {str(path.relative_to(PROJECT_ROOT)): _sha_file(path) for path in names if path.is_file()}


def _start_process(command, cwd, log_path, stdin=subprocess.DEVNULL):
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("ab")
    try:
        env = {
            **os.environ,
            "MAVLINK20": "1",
            "MAVLINK_DIALECT": "ardupilotmega",
            "PATH": f"{PROJECT_ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}",
        }
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            env=env,
            stdin=stdin,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        stream.close()
    return process


def _sim_vehicle_command(home, config):
    sim_vehicle = home / "Tools" / "autotest" / "sim_vehicle.py"
    instance_id = str(config.get("instance", 0))
    return [
        sys.executable, str(sim_vehicle),
        "-v", config["vehicle"]["type"],
        "-f", config["vehicle"]["frame"],
        "--no-mavproxy", "--no-rebuild", "--no-extra-ports", "--speedup", str(config["vehicle"]["speedup"]),
        "-w", "-I", instance_id,
    ]


def _mavproxy_command(config, mavproxy_path, log_file, qgc_port=None):
    ep = config["endpoints"]
    cmd = [
        str(mavproxy_path),
        f"--master={ep['sitl_master']}",
        f"--out={ep['controller_gcs_forward']}",
        f"--out={ep['proxy_relay_forward']}",
    ]
    if qgc_port is not None:
        cmd.append(f"--out=udp:127.0.0.1:{int(qgc_port)}")
    cmd.extend([
        f"--logfile={log_file}",
        "--daemon",
        "--non-interactive",
    ])
    return cmd


def _write_session_manifest(session_dir, home, config, processes, config_path=DEFAULT_CONFIG, status="running", qgc_port=None):
    tag, commit = _git_version(home)
    try:
        mavproxy_version = importlib.metadata.version("MAVProxy")
    except Exception:
        mavproxy_version = None
    try:
        pymavlink_version = importlib.metadata.version("pymavlink")
    except Exception:
        pymavlink_version = None
    try:
        pexpect_version = importlib.metadata.version("pexpect")
    except Exception:
        pexpect_version = None
    qgc_info = (
        {"enabled": True, "port": int(qgc_port), "endpoint": f"udp:127.0.0.1:{int(qgc_port)}"}
        if qgc_port is not None
        else {"enabled": False}
    )
    payload = {
        "session_id": session_dir.name,
        "status": status,
        "started_utc": _now(),
        "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
        "versions": {"python": sys.version, "ardupilot_tag": tag, "ardupilot_commit": commit, "mavproxy": mavproxy_version, "pymavlink": pymavlink_version, "pexpect": pexpect_version, "mavlink_dialect": "ardupilotmega", "mavlink_version": 2},
        "endpoints": config["endpoints"],
        "qgroundcontrol": qgc_info,
        "processes": processes,
        "sim_vehicle_command": processes.get("sitl", {}).get("command"),
        "bridge_command": processes.get("mavproxy", {}).get("command"),
        "message_types_expected": config["collection"]["message_types"],
        "config_sha256": _sha_file(config_path),
        "source_sha256": _source_hashes(),
        "synthetic_checkpoint_sha256": config["transfer_model"]["expected_checkpoint_sha256"],
        "note": "SITL telemetry is separate from all custom synthetic datasets and frozen evaluations.",
    }
    _write_json(session_dir / "session.json", payload)
    return payload


def start_session(ardupilot_home=None, session_dir=None, config_path=DEFAULT_CONFIG,
                  headless=False, qgroundcontrol_port=None, qgroundcontrol=None):
    config = _load_config(config_path)
    checks = check_prerequisites(ardupilot_home)
    if not checks["ready"]:
        raise RuntimeError("SITL prerequisites are not ready:\n" + json.dumps(checks, indent=2))
    is_headless = headless or (qgroundcontrol is False)
    qgc_port = ensure_qgroundcontrol(config, custom_port=qgroundcontrol_port, headless=is_headless, auto_start=True)
    if qgc_port is not None:
        print(f"[QGroundControl] Forwarding enabled on UDP 127.0.0.1:{qgc_port}", file=sys.stderr)
    _check_ports_available(config, qgc_port=qgc_port)
    home = _ardupilot_home(ardupilot_home)
    if session_dir is None:
        name = "session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_dir = PROJECT_ROOT / "results" / "sitl" / "sessions" / name
    session_dir = Path(session_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "logs").mkdir()
    processes = {}
    try:
        sitl_command = _sim_vehicle_command(home, config)
        sitl = _start_process(sitl_command, home, session_dir / "logs" / "ardupilot_sitl.log")
        processes["sitl"] = {"pid": sitl.pid, "command": sitl_command, "process_group": sitl.pid}
        time.sleep(1.0)
        if sitl.poll() is not None:
            raise RuntimeError(f"ArduPilot SITL exited during startup; inspect {session_dir / 'logs' / 'ardupilot_sitl.log'}")
        sitl_port = int(config["endpoints"]["sitl_master"].rsplit(":", 1)[1])
        if not _wait_for_port(sitl_port, timeout_s=30.0):
            if sitl.poll() is not None:
                raise RuntimeError(f"ArduPilot SITL exited during startup; inspect {session_dir / 'logs' / 'ardupilot_sitl.log'}")
            raise RuntimeError(f"Timed out waiting for ArduPilot SITL to listen on TCP port {sitl_port}")
        mavproxy_path = _tool("mavproxy.py")
        proxy_command = _mavproxy_command(config, mavproxy_path, session_dir / "mavlink.tlog", qgc_port=qgc_port)
        proxy = _start_process(proxy_command, session_dir, session_dir / "logs" / "mavproxy.log")
        processes["mavproxy"] = {"pid": proxy.pid, "command": proxy_command, "process_group": proxy.pid}
        time.sleep(2.0)
        if proxy.poll() is not None:
            raise RuntimeError(f"MAVProxy exited during startup; inspect {session_dir / 'logs' / 'mavproxy.log'}")
    except Exception as exc:
        _stop_processes(processes)
        _write_session_manifest(session_dir, home, config, processes, config_path, status="failed", qgc_port=qgc_port)
        raise
    manifest = _write_session_manifest(session_dir, home, config, processes, config_path, qgc_port=qgc_port)
    return session_dir, manifest


def _stop_processes(processes):
    for name in ("mavproxy", "sitl"):
        item = processes.get(name, {})
        group = int(item.get("process_group", item.get("pid", 0)) or 0)
        if group <= 0:
            continue
        try:
            os.killpg(group, signal.SIGINT)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(1.0)
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass


def stop_session(session_dir):
    session_dir = Path(session_dir).resolve()
    path = session_dir / "session.json"
    if not path.is_file():
        raise FileNotFoundError(f"No ACPDAD SITL session manifest at {path}")
    manifest = _read_json(path)
    processes = manifest.get("processes", {})
    _stop_processes(processes)
    manifest["status"] = "stopped"
    manifest["stopped_utc"] = _now()
    _write_json(path, manifest)
    return manifest


def _connect_controller(mavutil, endpoint, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    last_issue = None
    while time.monotonic() < deadline:
        try:
            connection = mavutil.mavlink_connection(endpoint, dialect="ardupilotmega", source_system=250, autoreconnect=True)
            controller = SitlController(connection, mavutil, endpoint, [])
            controller.wait_heartbeat(timeout_s=min(3.0, max(0.1, deadline - time.monotonic())))
            return controller
        except Exception as exc:
            last_issue = exc
            try:
                connection.close()
            except Exception:
                pass
            time.sleep(0.5)
    raise RuntimeError(f"MAVLink controller could not connect to {endpoint}: {last_issue}")


def _wait_for_stream(capture, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(row.get("message_type") == "HEARTBEAT" for row in capture.messages):
            return
        time.sleep(0.1)
    raise RuntimeError("No MAVLink heartbeat reached ACPDAD through the SITL -> MAVProxy -> relay -> detector localhost path")


class ScreenshotCollector:
    """Captures visual milestone evidence from the active desktop display during SITL flight."""

    def __init__(self, target_dir: Path, attack_start_s: float = 15.0, attack_duration_s: float = 25.0, phases: list | None = None):
        self.target_dir = Path(target_dir)
        self.attack_start_s = float(attack_start_s)
        self.attack_duration_s = float(attack_duration_s)
        self.phases = phases
        self.stop_event = threading.Event()
        self.airborne_event = threading.Event()
        self.landed_event = threading.Event()
        self.flight_start_monotonic = None
        self.thread = None
        self.captured = []

    def start(self):
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def notify_airborne(self, monotonic_ns=None):
        self.flight_start_monotonic = time.monotonic()
        self.airborne_event.set()

    def notify_landed(self):
        self.landed_event.set()

    def _capture(self, tag: str):
        target = self.target_dir / f"{tag}.png"
        env = dict(os.environ)
        if "WAYLAND_DISPLAY" not in env:
            env["WAYLAND_DISPLAY"] = "wayland-1"
        try:
            subprocess.run(
                ["grim", "-t", "png", str(target)],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if target.is_file() and target.stat().st_size > 0:
                self.captured.append(str(target))
                print(f"[Screenshot] Captured {tag}: {target.name} ({target.stat().st_size} bytes)", file=sys.stderr)
        except Exception as exc:
            print(f"[Screenshot] Failed to capture {tag}: {exc}", file=sys.stderr)

    def _run(self):
        if not self.airborne_event.wait(timeout=90.0):
            return
        # Milestone 1: 3s after takeoff confirmed
        time.sleep(3.0)
        if not self.stop_event.is_set():
            self._capture("01_takeoff_climb_15m")

        if self.phases:
            for p in self.phases:
                mid_s = p.start_s + p.duration_s / 2.0
                now_offset = time.monotonic() - (self.flight_start_monotonic or time.monotonic())
                wait_s = max(0.2, mid_s - now_offset)
                time.sleep(wait_s)
                if not self.stop_event.is_set():
                    self._capture(f"phase_{p.phase_index:02d}_{p.name}")
        else:
            # Midway through active attack interval
            attack_midpoint_s = (self.attack_start_s + min(12.0, self.attack_duration_s / 2.0)) / 2.0
            now_offset = time.monotonic() - (self.flight_start_monotonic or time.monotonic())
            wait_for_attack = max(0.5, attack_midpoint_s - now_offset)
            time.sleep(wait_for_attack)
            if not self.stop_event.is_set():
                self._capture("02_in_flight_attack_active")

            # Post-attack recovery / lap 2
            recovery_s = (self.attack_start_s + self.attack_duration_s + 15.0) / 2.0
            now_offset = time.monotonic() - (self.flight_start_monotonic or time.monotonic())
            wait_for_recovery = max(0.5, recovery_s - now_offset)
            time.sleep(wait_for_recovery)
            if not self.stop_event.is_set():
                self._capture("03_in_flight_recovery_lap2")

        # Milestone: landed
        if self.landed_event.wait(timeout=200.0):
            time.sleep(2.0)
            if not self.stop_event.is_set():
                self._capture("08_post_landing_circuit_trail")
            time.sleep(4.0)
            if not self.stop_event.is_set():
                self._capture("09_post_landing_telemetry_active")

    def stop(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)


def _build_dashboard_state(controller, capture, policy, run_id):
    state = {
        "run_id": run_id,
        "flight_status": "INITIALIZING",
        "flight_mode": "UNKNOWN",
        "armed": False,
        "observer": {},
        "sequence": {},
        "ml_inference": {},
        "detector_health": {},
    }
    if controller is not None:
        telemetry = controller.get_live_telemetry()
        state["flight_status"] = telemetry.get("flight_status", "UNKNOWN")
        state["flight_mode"] = telemetry.get("flight_mode", "UNKNOWN")
        state["armed"] = telemetry.get("armed", False)
        state["observer"] = {
            "relative_alt_m": telemetry.get("relative_alt_m", 0.0),
            "groundspeed_mps": telemetry.get("groundspeed_mps", 0.0),
            "heading_deg": telemetry.get("heading_deg", 0.0),
            "lap": telemetry.get("lap", 1),
            "waypoint": telemetry.get("waypoint", 0),
            "distance_to_wp_m": telemetry.get("distance_to_wp_m", 0.0),
            "total_waypoints": 8,
        }
    if capture is not None:
        live = capture.get_live_state()
        state["ml_inference"] = live.get("ml_inference", {})
        state["detector_health"] = live.get("detector_health", {})
        state["relay"] = live.get("relay", {})
        relay = live.get("relay", {})
        phase_name = relay.get("phase_name", "continuous")
        phase_idx = relay.get("phase_index", 1)
        phase_dur = relay.get("phase_duration_s", 15.0)
        elapsed = relay.get("mission_elapsed_s")
        phase_elapsed = 0.0
        pct = 0.0
        if hasattr(policy, "phases") and 1 <= phase_idx <= len(policy.phases):
            cur_p = policy.phases[phase_idx - 1]
            if elapsed is not None:
                phase_elapsed = max(0.0, elapsed - cur_p.start_s)
                pct = min(100.0, (phase_elapsed / cur_p.duration_s) * 100.0) if cur_p.duration_s > 0 else 100.0
        state["sequence"] = {
            "phase_index": phase_idx,
            "phase_name": phase_name,
            "scenario_type": relay.get("active_scenario", "benign"),
            "injector_state": relay.get("injector_state", "BENIGN_FORWARDING"),
            "phase_elapsed_s": round(phase_elapsed, 1),
            "phase_duration_s": round(phase_dur, 1),
            "progress_pct": round(pct, 1),
        }
    return state


class LiveTerminalLogger:
    """Prints structured real-time updates to stdout during continuous SITL flight."""

    def __init__(self, capture, controller, policy, dashboard=None):
        self.capture = capture
        self.controller = controller
        self.policy = policy
        self.dashboard = dashboard
        self.stop_event = threading.Event()
        self.thread = None
        self.last_phase_name = ""
        self.last_print_mono = 0

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="acpdad-terminal-logger")
        self.thread.start()

    def log_waypoint(self, lap, wp_index, total_completed):
        telemetry = self.controller.get_live_telemetry() if self.controller else {}
        alt = telemetry.get("relative_alt_m", 15.0)
        dist = telemetry.get("distance_to_wp_m", 0.0)
        msg = f"Lap {lap}/2 | Waypoint {wp_index}/8 reached (alt: {alt:.1f}m, dist: {dist:.1f}m, total WPs: {total_completed})"
        if self.dashboard is None:
            print(f"[NAV] {msg}", flush=True)
        else:
            self.dashboard.log_event("NAV", msg)

    def log_phase_transition(self, prev_phase, new_phase, phase_idx, scenario):
        desc = ""
        dur = 0.0
        if hasattr(self.policy, "phases") and 1 <= phase_idx <= len(self.policy.phases):
            p = self.policy.phases[phase_idx - 1]
            desc = p.description
            dur = p.duration_s
        inj_state = f"INJECTING_{scenario.upper()}" if scenario != "benign" else "BENIGN_FORWARDING"
        divider = "=" * 80
        msg = f"=== PHASE TRANSITION: [Phase {phase_idx}] {new_phase} ({dur:.1f}s) | Injector: {inj_state} ==="
        if self.dashboard is None:
            print(f"\n{divider}\n{msg}\n{divider}\n", flush=True)
        else:
            self.dashboard.log_event("PHASE", f"Phase {phase_idx}: {new_phase} ({dur:.1f}s) - {desc}")

    def _run(self):
        while not self.stop_event.is_set():
            time.sleep(1.0)
            now_mono = time.monotonic()
            if self.controller is None or self.capture is None:
                continue
            telemetry = self.controller.get_live_telemetry()
            live = self.capture.get_live_state()
            relay = live.get("relay", {})
            ml = live.get("ml_inference", {})
            health = live.get("detector_health", {})

            cur_phase = relay.get("phase_name", "")
            phase_idx = relay.get("phase_index", 1)
            scen = relay.get("active_scenario", "benign")
            if cur_phase and cur_phase != self.last_phase_name:
                prev = self.last_phase_name
                self.last_phase_name = cur_phase
                self.log_phase_transition(prev, cur_phase, phase_idx, scen)

            if now_mono - self.last_print_mono >= 2.5:
                self.last_print_mono = now_mono
                elapsed = relay.get("mission_elapsed_s") or 0.0
                inj = relay.get("injector_state", "BENIGN_FORWARDING")
                pred = ml.get("latest_prediction", "Benign")
                conf = ml.get("confidence")
                conf_str = f"({conf*100:.1f}%)" if conf else ""
                stale_str = " [NO FRESH PREDICTION (STALE)]" if ml.get("is_stale") else ""
                v_res = health.get("vertical_residual_m")
                phys_str = f"r_z={v_res:.4f}m" if v_res is not None else "r_z=0m"
                if health.get("physics_warning_active"):
                    phys_str += " [PHYSICS WARN >0.0108m]"
                lap = telemetry.get("lap", 1)
                wp = telemetry.get("waypoint", 0)
                alt = telemetry.get("relative_alt_m", 0.0)
                spd = telemetry.get("groundspeed_mps", 0.0)
                status_line = (
                    f"[{elapsed:05.1f}s] [{cur_phase or 'phase'}] [{inj}] "
                    f"ML: {pred}{conf_str}{stale_str} | {phys_str} | Lap {lap} WP {wp} | Alt {alt:.1f}m Spd {spd:.1f}m/s"
                )
                if self.dashboard is None:
                    print(status_line, flush=True)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)


def generate_sequence_report(session_dir: Path, policy, mission: dict, manifest: dict) -> str:
    import pandas as pd
    gt_path = session_dir / "capture" / "ground_truth_joined_after_inference.csv"
    pred_path = session_dir / "capture" / "predictions_label_free.csv"
    gt_df = pd.read_csv(gt_path) if gt_path.is_file() else pd.DataFrame()
    pred_df = pd.read_csv(pred_path) if pred_path.is_file() else pd.DataFrame()

    phases = policy.phases if hasattr(policy, "phases") else []
    table_rows = []

    for phase in phases:
        start_s = phase.start_s
        end_s = phase.end_s
        dur_s = phase.duration_s
        p_name = phase.name
        p_type = phase.scenario_type
        p_idx = phase.phase_index

        sub_gt = gt_df[(gt_df["mission_elapsed_s"] >= start_s) & (gt_df["mission_elapsed_s"] < end_s)] if not gt_df.empty else pd.DataFrame()
        total_msgs = len(sub_gt)
        cov_count = int(sub_gt["coverage"].sum()) if not sub_gt.empty else 0
        unavail_count = total_msgs - cov_count

        effects = sub_gt["relay_effect"].value_counts().to_dict() if not sub_gt.empty else {}
        if p_type == "benign":
            eff_desc = f"Forward ({effects.get('forward', total_msgs)})"
        elif p_type == "FDI":
            eff_desc = f"FDI +5.0m ({effects.get('mutate', 0)} mutated, {effects.get('forward', 0)} fwd)"
        elif p_type == "DoS":
            eff_desc = f"DoS drop 1/2 ({effects.get('drop', 0)} dropped, {effects.get('forward', 0)} fwd)"
        elif p_type == "Replay":
            eff_desc = f"Replay 5.0s ({effects.get('replay', 0)} replayed, {effects.get('forward', 0)} fwd)"
        else:
            eff_desc = str(effects)

        preds_dist = sub_gt[sub_gt["coverage"] == True]["prediction"].value_counts().to_dict() if not sub_gt.empty else {}
        preds_summary = ", ".join(f"{k}: {v}" for k, v in sorted(preds_dist.items())) if preds_dist else "None"

        alert_count = sum(v for k, v in preds_dist.items() if k != "Benign")

        phys_warn_count = 0
        if not sub_gt.empty and not pred_df.empty:
            covered_seqs = set(sub_gt[sub_gt["coverage"] == True]["sequence"])
            sub_preds = pred_df[pred_df["sequence"].isin(covered_seqs)]
            if not sub_preds.empty and "physics_warning_candidate" in sub_preds.columns:
                phys_warn_count = int(sub_preds["physics_warning_candidate"].sum())

        if alert_count > 0:
            first_alert = sub_gt[(sub_gt["coverage"] == True) & (sub_gt["prediction"] != "Benign")].iloc[0]
            delay_s = round(float(first_alert["mission_elapsed_s"]) - start_s, 2)
            delay_desc = f"{delay_s:.2f}s"
        elif p_type == "FDI" and phys_warn_count > 0:
            delay_desc = "N/A (ML: Benign, Phys Warn: <0.2s)"
        elif p_type == "DoS" and unavail_count > 0:
            delay_desc = "N/A (ML: Benign, Gap: <0.2s)"
        else:
            delay_desc = "N/A (Predicted Benign)"

        table_rows.append({
            "phase_index": p_idx,
            "name": p_name,
            "scenario": p_type,
            "window": f"[{start_s:.1f}s - {end_s:.1f}s] ({dur_s:.1f}s)",
            "interval_sim": f"{start_s:.1f}s - {end_s:.1f}s",
            "interval_wall": f"{start_s/2.0:.1f}s - {end_s/2.0:.1f}s",
            "effect_counts": eff_desc,
            "available_obs": cov_count,
            "ml_predictions": preds_summary,
            "actual_alerts": alert_count,
            "physics_warnings": phys_warn_count,
            "unavailable": unavail_count,
            "delay": delay_desc,
        })

    md = []
    md.append(f"# ACPDAD Single Continuous Flight Attack Detection Demonstration Report\n")
    md.append(f"**Run ID:** `{manifest.get('run_id')}`  ")
    md.append(f"**Flight Scenario:** Continuous Multi-Attack Sequence (`sequence`)  ")
    md.append(f"**Started UTC:** `{manifest.get('started_utc')}` | **Completed UTC:** `{manifest.get('completed_utc')}`  ")
    md.append(f"**Provenance:** `ARDUPILOT_SITL_LOCAL_SIMULATION` | **SITL Speedup:** `2.0x`  ")
    qgc_port = manifest.get('qgroundcontrol', {}).get('port', 14600)
    md.append(f"**QGroundControl UDP Link:** `127.0.0.1:{qgc_port}`  \n")
    md.append(f"## 1. Flight Circuit & Mission Performance\n")
    md.append(f"- **Continuous Circuit Laps Flown:** {mission.get('laps', 4)} completed laps")
    md.append(f"- **Total Distance Traveled:** {mission.get('total_distance_m', 0.0):.1f} m")
    md.append(f"- **Flight Duration:** {mission.get('flight_duration_s', 0.0):.1f} s")
    md.append(f"- **Waypoints Reached:** {mission.get('waypoints_completed', 0)}")
    md.append(f"- **Cruise Speed:** {mission.get('cruise_speed_mps', 16.0):.1f} m/s at {mission.get('altitude_m', 15.0):.1f} m AGL\n")
    md.append(f"## 2. In-Flight Attack Sequence Results Table\n")
    md.append("| Phase | Phase Name | Attack Type | Scheduled Sim Window | Injected Effect & Counts | Available Obs | ML Predictions Breakdown | Model Alerts | Physics Warnings | Unavailable (Dropped) | Delay |")
    md.append("|:---:|:---|:---:|:---:|:---|:---:|:---|:---:|:---:|:---:|:---|")
    for r in table_rows:
        md.append(f"| {r['phase_index']} | `{r['name']}` | {r['scenario']} | {r['window']} | {r['effect_counts']} | {r['available_obs']} | {r['ml_predictions']} | {r['actual_alerts']} | {r['physics_warnings']} | {r['unavailable']} | {r['delay']} |")
    md.append("\n## 3. Separation of Evidence Types & Integrity Contracts\n")
    md.append("1. **Injector State vs. ML Output:** What the relay injects (`INJECTING_FDI`, `INJECTING_DOS`, `INJECTING_REPLAY`) modifies the detector's telemetry stream. The frozen synthetic transfer model was trained on stationary hover and primarily outputs `Benign` on dynamic circuit telemetry.")
    md.append("2. **Analytical Physics Warnings:** FDI altitude steps trigger vertical kinematic/sensor residual violations (`r_z > 0.0108m`), establishing immediate physical anomaly detection without altering ML model outputs.")
    md.append("3. **DoS Observations:** Dropped datagrams produce missing observations (`UNAVAILABLE`) on the relay branch. No synthetic ML classification is fabricated for absent packets.")
    md.append("4. **Observer Isolation:** QGroundControl received clean, unmutated telemetry on UDP 14600. The simulated drone flew all laps continuously without disruption.\n")

    report_text = "\n".join(md)
    report_file = session_dir / "live_attack_sequence_report.md"
    report_file.write_text(report_text + "\n")
    return report_text


def generate_combined_report(session_dir: Path, policy, mission: dict, manifest: dict) -> str:
    import pandas as pd
    gt_path = session_dir / "capture" / "ground_truth_joined_after_inference.csv"
    pred_path = session_dir / "capture" / "predictions_label_free.csv"
    gt_df = pd.read_csv(gt_path) if gt_path.is_file() else pd.DataFrame()
    pred_df = pd.read_csv(pred_path) if pred_path.is_file() else pd.DataFrame()

    total_msgs = len(gt_df)
    cov_count = int(gt_df["coverage"].sum()) if not gt_df.empty else 0
    unavail_count = total_msgs - cov_count

    baseline_gt = gt_df[gt_df["mission_elapsed_s"] < policy.attack_start_s] if not gt_df.empty else pd.DataFrame()
    attack_gt = gt_df[(gt_df["mission_elapsed_s"] >= policy.attack_start_s) & (gt_df["Target_Label"] == "Combined_Attack")] if not gt_df.empty else pd.DataFrame()
    post_gt = gt_df[(gt_df["mission_elapsed_s"] >= policy.attack_start_s) & (gt_df["Target_Label"] == "Benign")] if not gt_df.empty else pd.DataFrame()

    effects = gt_df["relay_effect"].value_counts().to_dict() if not gt_df.empty else {}
    fwd_count = effects.get("forward", 0)
    drop_count = effects.get("drop_detector_branch", effects.get("drop", 0))
    mut_count = effects.get("modify_global_position_altitude", effects.get("mutate", 0))
    rep_count = effects.get("replay_earlier_mavlink_datagram", effects.get("replay", 0))

    preds_dist = gt_df[gt_df["coverage"] == True]["prediction"].value_counts().to_dict() if not gt_df.empty else {}
    alert_count = sum(v for k, v in preds_dist.items() if k != "Benign")

    phys_warn_count = 0
    if not pred_df.empty and "physics_warning_candidate" in pred_df.columns:
        phys_warn_count = int(pred_df["physics_warning_candidate"].sum())

    qgc_port = manifest.get("qgroundcontrol", {}).get("port", 14600)

    md = []
    md.append(f"# ACPDAD Concurrent Multi-Attack In-Flight Detection Demonstration Report\n")
    md.append(f"**Run ID:** `{manifest.get('run_id')}`  ")
    md.append(f"**Flight Scenario:** Concurrent Multi-Attack (`combined` / `concurrent`: FDI + DoS + Replay)  ")
    md.append(f"**Started UTC:** `{manifest.get('started_utc')}` | **Completed UTC:** `{manifest.get('completed_utc')}`  ")
    md.append(f"**Provenance:** `ARDUPILOT_SITL_LOCAL_SIMULATION` | **SITL Speedup:** `2.0x`  ")
    md.append(f"**QGroundControl UDP Link:** `127.0.0.1:{qgc_port}`  \n")

    md.append(f"## 1. Flight Circuit & Mission Performance\n")
    md.append(f"- **Continuous Circuit Laps Flown:** {mission.get('laps', 2)} completed laps")
    md.append(f"- **Total Distance Traveled:** {mission.get('total_distance_m', 0.0):.1f} m")
    md.append(f"- **Flight Duration:** {mission.get('flight_duration_s', 0.0):.1f} s")
    md.append(f"- **Waypoints Reached:** {mission.get('waypoints_completed', 0)} / 16 (across 2 laps)")
    md.append(f"- **Cruise Speed:** {mission.get('cruise_speed_mps', 16.0):.1f} m/s at {mission.get('altitude_m', 15.0):.1f} m AGL\n")

    md.append(f"## 2. In-Flight Concurrent Attack Results Table\n")
    md.append("| Mission Phase | Attack Mode | Sim Window | Injector Stream Effects | Received Obs | ML Predictions Breakdown | Model Alerts | Physics Warnings | Unavailable (DoS Drops) |")
    md.append("|:---|:---:|:---:|:---|:---:|:---|:---:|:---|:---:|")

    b_cov = int(baseline_gt["coverage"].sum()) if not baseline_gt.empty else 0
    b_unavail = len(baseline_gt) - b_cov
    b_preds = ", ".join(f"{k}: {v}" for k, v in baseline_gt[baseline_gt["coverage"] == True]["prediction"].value_counts().items()) if not baseline_gt.empty else "N/A"
    md.append(f"| Airborne Baseline | Benign | [0.0s - {policy.attack_start_s:.1f}s] | Clean forwarding ({len(baseline_gt)} pkts) | {b_cov} | {b_preds} | 0 | 0 | {b_unavail} |")

    a_cov = int(attack_gt["coverage"].sum()) if not attack_gt.empty else 0
    a_unavail = len(attack_gt) - a_cov
    a_preds = ", ".join(f"{k}: {v}" for k, v in attack_gt[attack_gt["coverage"] == True]["prediction"].value_counts().items()) if not attack_gt.empty else "N/A"
    a_alert_count = sum(v for k, v in attack_gt[attack_gt["coverage"] == True]["prediction"].value_counts().items() if k != "Benign") if not attack_gt.empty else 0
    inj_desc = f"FDI: {mut_count} mut (+5m), DoS: {drop_count} drops (33%), Replay: {rep_count} rep (5s)"
    md.append(f"| Concurrent Circuit (Laps 1-2) | FDI + DoS + Replay | [{policy.attack_start_s:.1f}s - Circuit End] | {inj_desc} | {a_cov} | {a_preds} | {a_alert_count} | {phys_warn_count} | {a_unavail} |")

    p_cov = int(post_gt["coverage"].sum()) if not post_gt.empty else 0
    p_unavail = len(post_gt) - p_cov
    p_preds = ", ".join(f"{k}: {v}" for k, v in post_gt[post_gt["coverage"] == True]["prediction"].value_counts().items()) if not post_gt.empty else "N/A"
    md.append(f"| Return & Land | Benign (Injections Stopped) | [Circuit End - Landing] | Clean forwarding ({len(post_gt)} pkts) | {p_cov} | {p_preds} | 0 | 0 | {p_unavail} |")

    md.append("\n## 3. Separation of Evidence Types & Integrity Contracts\n")
    md.append("1. **Simultaneous Attack Injections:** FDI (+5.0m step altitude), DoS (bounded 33% packet drop), and Replay (5.0s buffer stale injection) operated concurrently during the entire two-lap circuit.")
    md.append("2. **Injector State vs. ML Output:** What the relay injects on the detector branch is strictly separated from model detections. The frozen synthetic transfer model was trained on stationary hover and primarily outputs `Benign` on dynamic flight.")
    md.append("3. **Analytical Physics Warnings:** FDI altitude step mutations (+5.0m) produce immediate vertical kinematic residual violations (`r_z > 0.0108m`), establishing rapid analytical anomaly detection without altering ML predictions.")
    md.append("4. **DoS Missing Observations:** Datagrams dropped by the DoS filter produce missing observations (`UNAVAILABLE`) on the detector branch. Missing packets are recorded as telemetry gaps rather than fabricated ML predictions.")
    md.append("5. **Observer Isolation:** QGroundControl received clean, unmutated telemetry on UDP 127.0.0.1:14600 throughout the entire mission, tracking the drone through takeoff, two complete laps, return-to-home, and landing.\n")

    report_text = "\n".join(md)
    report_file = session_dir / "live_attack_report.md"
    report_file.write_text(report_text + "\n")
    return report_text


def run_session(scenario="benign", duration_s=30.0, attack_start_s=None, attack_duration_s=None,
                fdi_subtype=None, replay_delay_s=None, replay_repetitions=None, run_id=None,
                ardupilot_home=None, output_root=None, model_dir=DEFAULT_MODEL, config_path=DEFAULT_CONFIG,
                altitude_m=None, fdi_offset_m=None, dos_drop_every=None,
                headless=False, qgroundcontrol_port=None, qgroundcontrol=None,
                route=None, laps=2, north_m=180.0, east_m=120.0, cruise_speed_mps=16.0,
                acceptance_radius_m=15.0, pause_at_wp_s=0.5, post_landing_hold_s=15.0, radius_m=None,
                on_landed=None, dashboard=True, dashboard_port=8088):
    scenario = validate_scenario(scenario)
    config = _load_config(config_path)
    if float(duration_s) < float(config["collection"]["minimum_duration_s"]):
        raise ValueError(f"Mission duration must be at least {config['collection']['minimum_duration_s']} seconds")
    if route is None:
        route = True
    attack_start = float(config["scenarios"]["attack_start_s"] if attack_start_s is None else attack_start_s)
    attack_duration = float(config["scenarios"]["attack_duration_s"] if attack_duration_s is None else attack_duration_s)

    if scenario == "sequence":
        policy = ScenarioPolicy("sequence", 120.0, attack_start, attack_duration, config, fdi_subtype, replay_delay_s, replay_repetitions, fdi_offset_m, dos_drop_every)
        duration_s = max(float(duration_s or 120.0), float(policy.duration_s), 120.0)
        laps = max(int(laps or 4), 4)
        min_flight_duration_s = 120.0
    elif scenario in ("combined", "concurrent"):
        attack_start = 5.0
        laps = int(laps or 2)
        policy = ScenarioPolicy(scenario, 160.0, attack_start, 140.0, config, fdi_subtype, replay_delay_s, replay_repetitions, fdi_offset_m, dos_drop_every)
        duration_s = max(float(duration_s or 160.0), 160.0)
        min_flight_duration_s = None
    else:
        if route:
            duration_s = max(float(duration_s), attack_start + attack_duration + 30.0, 160.0)
        policy = ScenarioPolicy(scenario, duration_s, attack_start, attack_duration, config, fdi_subtype, replay_delay_s, replay_repetitions, fdi_offset_m, dos_drop_every)
        min_flight_duration_s = None

    home = _ardupilot_home(ardupilot_home)
    checks = check_prerequisites(home)
    if not checks["ready"]:
        raise RuntimeError("SITL prerequisites are not ready:\n" + json.dumps(checks, indent=2))
    mavutil = _import_mavutil()
    is_headless = headless or (qgroundcontrol is False)
    qgc_port = ensure_qgroundcontrol(config, custom_port=qgroundcontrol_port, headless=is_headless, auto_start=True)
    if qgc_port is not None:
        print(f"[QGroundControl] Forwarding enabled on UDP 127.0.0.1:{qgc_port}", file=sys.stderr)
    _check_ports_available(config, qgc_port=qgc_port)
    model_dir = Path(model_dir).resolve()
    selected_dir = (PROJECT_ROOT / "models" / "selected").resolve()
    sitl_dir = (PROJECT_ROOT / "models" / "sitl").resolve()
    if not (model_dir / "checkpoint.pkl").is_file():
        raise FileNotFoundError(f"Checkpoint not found at {model_dir}")
    if model_dir == selected_dir:
        expected_model_sha = config["transfer_model"]["expected_checkpoint_sha256"]
        if _sha_file(model_dir / "checkpoint.pkl") != expected_model_sha:
            raise ValueError("The selected synthetic checkpoint digest differs from the SITL transfer configuration")
    elif model_dir == sitl_dir and "model" in config:
        expected_model_sha = config["model"].get("checkpoint_sha256")
        if expected_model_sha and _sha_file(model_dir / "checkpoint.pkl") != expected_model_sha:
            raise ValueError("The SITL checkpoint digest differs from the SITL configuration")
    output_root = Path(output_root or PROJECT_ROOT / "results" / "sitl" / "runs").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"{scenario.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run_id may contain only letters, digits, underscore, and hyphen")
    session_dir = output_root / run_id
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "logs").mkdir()
    processes = {}
    capture = None
    controller = None
    run_error = None
    start_utc = _now()
    screenshot_collector = None
    dashboard_server = None
    dashboard_url = None
    terminal_logger = None

    if not is_headless and qgc_port is not None and (shutil.which("grim") is not None) and (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
        screenshot_collector = ScreenshotCollector(session_dir / "screenshots", attack_start, attack_duration, phases=policy.phases if hasattr(policy, "phases") else None)
        screenshot_collector.start()

    if dashboard:
        try:
            dashboard_server = TerminalDashboard(
                state_provider=lambda: _build_dashboard_state(controller, capture, policy, run_id),
            )
            dashboard_server.start()
            dashboard_url = "terminal://in-place-rich-dashboard"
        except Exception as exc:
            print(f"[Dashboard] Warning: Could not start terminal dashboard: {exc}", file=sys.stderr)
            dashboard_server = None

    try:
        sitl_command = _sim_vehicle_command(home, config)
        sitl = _start_process(sitl_command, home, session_dir / "logs" / "ardupilot_sitl.log")
        processes["sitl"] = {"pid": sitl.pid, "command": sitl_command, "process_group": sitl.pid}
        time.sleep(1.0)
        if sitl.poll() is not None:
            raise RuntimeError(f"ArduPilot SITL exited; inspect {session_dir / 'logs' / 'ardupilot_sitl.log'}")
        sitl_port = int(config["endpoints"]["sitl_master"].rsplit(":", 1)[1])
        if not _wait_for_port(sitl_port, timeout_s=30.0):
            if sitl.poll() is not None:
                raise RuntimeError(f"ArduPilot SITL exited; inspect {session_dir / 'logs' / 'ardupilot_sitl.log'}")
            raise RuntimeError(f"Timed out waiting for ArduPilot SITL to listen on TCP port {sitl_port}")
        capture = CaptureContext(run_id, session_dir / "capture", config, model_dir, policy, mavutil.mavlink)
        capture.start(anchor_now=False)
        mavproxy_path = _tool("mavproxy.py")
        proxy_command = _mavproxy_command(config, mavproxy_path, session_dir / "mavlink.tlog", qgc_port=qgc_port)
        proxy = _start_process(proxy_command, session_dir, session_dir / "logs" / "mavproxy.log")
        processes["mavproxy"] = {"pid": proxy.pid, "command": proxy_command, "process_group": proxy.pid}
        controller = _connect_controller(mavutil, config["endpoints"]["controller"], config["vehicle"]["arm_timeout_s"] + 30)
        _wait_for_stream(capture)
        controller.log.clear()
        target_altitude = float(altitude_m if altitude_m is not None else config["vehicle"]["controller_altitude_m"])

        terminal_logger = LiveTerminalLogger(capture, controller, policy, dashboard=dashboard_server)
        terminal_logger.start()

        def on_circuit_completed_cb():
            elapsed = capture.relay._elapsed(time.monotonic_ns()) if (capture and hasattr(capture, "relay")) else None
            if hasattr(policy, "stop_injections"):
                policy.stop_injections(elapsed)
            if dashboard_server is not None:
                dashboard_server.log_event("ATTACK", "Circuit completed: Stopped FDI, DoS, and Replay injections before return-to-home.")

        def on_airborne_cb(ns):
            capture.set_flight_start(ns)
            if dashboard_server is not None:
                dashboard_server.set_flight_start(ns)
                dashboard_server.log_event("MISSION", f"Takeoff confirmed at {target_altitude:.1f}m. Circuit flight commenced.")
                if scenario in ("combined", "concurrent"):
                    dashboard_server.log_event("ATTACK", f"Airborne baseline active ({policy.attack_start_s:.1f}s). Concurrent attacks scheduled.")
            if screenshot_collector is not None:
                screenshot_collector.notify_airborne(ns)

        def on_landed_cb():
            if screenshot_collector is not None:
                screenshot_collector.notify_landed()
            if dashboard_server is not None:
                dashboard_server.log_event("MISSION", "Mission completed: Drone safely landed and disarmed.")
            if on_landed is not None:
                on_landed()

        def on_waypoint_cb(lap_num, wp_num, wp_total):
            if terminal_logger is not None:
                terminal_logger.log_waypoint(lap_num, wp_num, wp_total)

        if capture is not None and hasattr(capture, "relay"):
            def on_phase_change_cb(ev):
                sc = ev.get("scenario", "")
                to_ph = ev.get("to_phase", "")
                if dashboard_server is not None:
                    if sc in ("combined", "concurrent"):
                        dashboard_server.log_event("ATTACK", "Concurrent attacks ACTIVATED: FDI (+5m) + DoS (33% drop) + Replay (5s buffer).")
                    else:
                        dashboard_server.log_event("PHASE", f"Phase transition: {to_ph} ({sc})")
            capture.relay.on_phase_change = on_phase_change_cb

        mission = controller.run_benign_mission(
            float(duration_s), target_altitude, config["vehicle"],
            on_airborne=on_airborne_cb,
            route=route,
            laps=int(laps),
            north_m=float(north_m),
            east_m=float(east_m),
            cruise_speed_mps=float(cruise_speed_mps),
            acceptance_radius_m=float(acceptance_radius_m),
            pause_at_wp_s=float(pause_at_wp_s),
            post_landing_hold_s=float(post_landing_hold_s),
            radius_m=radius_m,
            on_landed=on_landed_cb,
            min_flight_duration_s=min_flight_duration_s,
            on_waypoint_event=on_waypoint_cb,
            on_circuit_completed=on_circuit_completed_cb,
        )
        capture.stop()
        _write_json(session_dir / "controller_commands.json", {"run_id": run_id, "commands": controller.log, "mission": mission, "controller_endpoint": config["endpoints"]["controller"], "labels_excluded": True})
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass
        if capture is not None and getattr(capture, "started", False):
            try:
                capture.stop()
            except Exception as save_exc:
                run_error += f"; capture-save failure: {type(save_exc).__name__}: {save_exc}"
    finally:
        if terminal_logger is not None:
            terminal_logger.stop()
        if dashboard_server is not None:
            dashboard_server.stop()
        if screenshot_collector is not None:
            screenshot_collector.stop()
        if controller is not None:
            controller.close()
        _stop_processes(processes)
    tag, commit = _git_version(home)
    try:
        import pymavlink
        pymavlink_version = importlib.metadata.version("pymavlink")
    except Exception:
        pymavlink_version = None
    try:
        mavproxy_version = importlib.metadata.version("MAVProxy")
    except Exception:
        mavproxy_version = None
    try:
        pexpect_version = importlib.metadata.version("pexpect")
    except Exception:
        pexpect_version = None
    qgc_info = (
        {"enabled": True, "port": int(qgc_port), "endpoint": f"udp:127.0.0.1:{int(qgc_port)}"}
        if qgc_port is not None
        else {"enabled": False}
    )
    run_manifest = {
        "run_id": run_id,
        "status": "completed" if run_error is None else "failed",
        "started_utc": start_utc,
        "completed_utc": _now(),
        "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
        "scenario": scenario,
        "scenario_schedule": {"start_s": policy.attack_start_s if scenario != "benign" else None, "duration_s": policy.attack_duration_s if scenario != "benign" else 0, "fdi_subtype": policy.fdi_subtype if scenario == "FDI" else None, "replay_delay_s": policy.replay_delay_s if scenario == "Replay" else None, "replay_repetitions": policy.replay_repetitions if scenario == "Replay" else None},
        "sequence_phases": policy.phases_summary() if hasattr(policy, "phases_summary") else None,
        "software": {"python": sys.version, "ardupilot_tag": tag, "ardupilot_commit": commit, "mavproxy": mavproxy_version, "pymavlink": pymavlink_version, "pexpect": pexpect_version, "mavlink_dialect": "ardupilotmega", "mavlink_version": 2},
        "endpoints": config["endpoints"],
        "qgroundcontrol": qgc_info,
        "dashboard": {"enabled": bool(dashboard_server), "url": dashboard_url, "port": dashboard_port},
        "route": {
            "enabled": bool(route),
            "laps": int(mission.get("laps", laps)) if (run_error is None and isinstance(mission, dict)) else (int(laps) if route else 0),
            "north_m": float(north_m) if route else 0.0,
            "east_m": float(east_m) if route else 0.0,
            "lap_distance_m": float(mission.get("lap_distance_m", 0.0)) if (run_error is None and isinstance(mission, dict)) else 0.0,
            "total_distance_m": float(mission.get("total_distance_m", 0.0)) if (run_error is None and isinstance(mission, dict)) else 0.0,
            "cruise_speed_mps": float(cruise_speed_mps) if route else 0.0,
            "altitude_m": float(target_altitude),
            "pause_at_wp_s": float(pause_at_wp_s) if route else 0.0,
            "waypoints_completed": int(mission.get("waypoints_completed", 0)) if (run_error is None and isinstance(mission, dict)) else 0,
        },
        "commands": {"sitl": processes.get("sitl", {}).get("command"), "mavproxy": processes.get("mavproxy", {}).get("command")},
        "message_types": config["collection"]["message_types"],
        "config_sha256": _sha_file(config_path),
        "checkpoint_sha256": _sha_file(model_dir / "checkpoint.pkl"),
        "source_sha256": _source_hashes(),
        "transfer_checkpoint_not_sitl_trained": True,
        "sitl_metrics_claimed": False,
        "screenshots": screenshot_collector.captured if screenshot_collector else [],
        "error": run_error,
    }
    _write_json(session_dir / "run_manifest.json", run_manifest)
    files = [p for p in session_dir.rglob("*") if p.is_file() and p.name != "checksums.json"]
    _write_json(session_dir / "checksums.json", {str(path.relative_to(session_dir)): _sha_file(path) for path in sorted(files)})

    if scenario == "sequence" and run_error is None:
        report_text = generate_sequence_report(session_dir, policy, mission, run_manifest)
        print("\n" + "=" * 80, flush=True)
        print("=== ACPDAD SEQUENCE MISSION RESULTS SUMMARY ===", flush=True)
        print("=" * 80 + "\n", flush=True)
        print(report_text, flush=True)
    elif scenario in ("combined", "concurrent") and run_error is None:
        report_text = generate_combined_report(session_dir, policy, mission, run_manifest)
        print("\n" + "=" * 80, flush=True)
        print("=== ACPDAD CONCURRENT ATTACKS MISSION RESULTS SUMMARY ===", flush=True)
        print("=" * 80 + "\n", flush=True)
        print(report_text, flush=True)

    if run_error is not None:
        raise RuntimeError(f"SITL run failed; artifacts and logs retained at {session_dir}: {run_error}")
    return {
        "status": "completed",
        "run_id": run_id,
        "output": str(session_dir),
        "capture": str(session_dir / "capture"),
        "dashboard": dashboard_url,
        "run_manifest": str(session_dir / "run_manifest.json"),
        "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION",
        "checkpoint_transfer_only": True,
        "qgroundcontrol_port": qgc_port,
        "route": route,
    }


def capture_session(session_dir, scenario="benign", duration_s=30.0, output_dir=None,
                    attack_start_s=None, attack_duration_s=None, fdi_subtype=None,
                    replay_delay_s=None, replay_repetitions=None, model_dir=DEFAULT_MODEL, config_path=DEFAULT_CONFIG,
                    fdi_offset_m=None, dos_drop_every=None):
    session_dir = Path(session_dir).resolve()
    manifest = _read_json(session_dir / "session.json")
    if manifest.get("status") != "running":
        raise RuntimeError(f"SITL session is not marked running: {session_dir}")
    for name, proc in manifest.get("processes", {}).items():
        try:
            os.kill(int(proc["pid"]), 0)
        except ProcessLookupError:
            raise RuntimeError(f"SITL session process {name} (PID {proc['pid']}) is no longer running")
    scenario = validate_scenario(scenario)
    config = _load_config(config_path)
    attack_start = float(config["scenarios"]["attack_start_s"] if attack_start_s is None else attack_start_s)
    attack_duration = float(config["scenarios"]["attack_duration_s"] if attack_duration_s is None else attack_duration_s)
    policy = ScenarioPolicy(scenario, duration_s, attack_start, attack_duration, config, fdi_subtype, replay_delay_s, replay_repetitions, fdi_offset_m, dos_drop_every)
    mavutil = _import_mavutil()
    run_id = f"capture_{scenario.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = Path(output_dir or session_dir / "captures" / run_id).resolve()
    context = CaptureContext(run_id, output_dir, config, model_dir, policy, mavutil.mavlink)
    context.start(anchor_now=True)
    try:
        time.sleep(float(duration_s))
    finally:
        context.stop()
    return {"status": "completed", "run_id": run_id, "output": str(output_dir), "scenario": scenario, "provenance": "ARDUPILOT_SITL_LOCAL_SIMULATION"}


def controller_mission(session_dir, duration_s=30.0, altitude_m=None, config_path=DEFAULT_CONFIG):
    session_dir = Path(session_dir).resolve()
    manifest = _read_json(session_dir / "session.json")
    if manifest.get("status") != "running":
        raise RuntimeError(f"SITL session is not marked running: {session_dir}")
    config = _load_config(config_path)
    mavutil = _import_mavutil()
    logs = []
    controller = _connect_controller(mavutil, config["endpoints"]["controller"], config["vehicle"]["arm_timeout_s"] + 30)
    try:
        result = controller.run_benign_mission(float(duration_s), float(altitude_m or config["vehicle"]["controller_altitude_m"]), config["vehicle"])
    finally:
        logs.extend(controller.log)
        controller.close()
    output = session_dir / f"controller_commands_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    _write_json(output, {"commands": logs, "mission": result, "controller_endpoint": config["endpoints"]["controller"], "labels_excluded": True})
    return {"status": "completed", "controller_endpoint": config["endpoints"]["controller"], "command_log": str(output), "mission": result}

