"""MAVLink controller for a conservative SITL-only takeoff/land mission."""

from __future__ import annotations

import math
import time


class ControllerError(RuntimeError):
    pass


class SitlController:
    def __init__(self, connection, mavutil, endpoint: str, log: list[dict]):
        self.connection = connection
        self.mavutil = mavutil
        self.endpoint = endpoint
        self.log = log
        self.target_system = 0
        self.target_component = 0
        self.live_telemetry = {
            "flight_status": "INITIALIZING",
            "flight_mode": "UNKNOWN",
            "armed": False,
            "relative_alt_m": 0.0,
            "lat": 0,
            "lon": 0,
            "groundspeed_mps": 0.0,
            "heading_deg": 0.0,
            "lap": 0,
            "waypoint": 0,
            "total_waypoints": 8,
            "last_update_ns": 0,
        }

    def get_live_telemetry(self) -> dict:
        return dict(self.live_telemetry)

    def _record(self, action, **fields):
        self.log.append({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action": action, **fields})

    def wait_heartbeat(self, timeout_s=60.0):
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            message = self.connection.recv_match(type="HEARTBEAT", blocking=True, timeout=min(1.0, deadline - time.monotonic()))
            if message is not None:
                self.target_system = int(self.connection.target_system)
                self.target_component = int(self.connection.target_component)
                flag = int(self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.live_telemetry["armed"] = bool(int(message.base_mode) & flag)
                self.live_telemetry["last_update_ns"] = time.monotonic_ns()
                self._record("heartbeat", system_id=self.target_system, component_id=self.target_component, autopilot=int(message.autopilot), vehicle_type=int(message.type))
                return message
        raise ControllerError(f"No heartbeat received from SITL on {self.endpoint} within {timeout_s:g}s")

    def _wait_ack(self, command, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=min(1.0, deadline - time.monotonic()))
            if message is not None and int(message.command) == int(command):
                result = int(message.result)
                self._record("command_ack", command=int(command), result=result)
                accepted = int(self.mavutil.mavlink.MAV_RESULT_ACCEPTED)
                in_progress = int(getattr(self.mavutil.mavlink, "MAV_RESULT_IN_PROGRESS", 5))
                if result not in (accepted, in_progress):
                    raise ControllerError(f"SITL rejected MAVLink command {command}: result={result}")
                return result
        raise ControllerError(f"Timed out waiting for COMMAND_ACK for command {command}")

    def _command_long(self, command, params, timeout_s=10.0):
        values = list(params) + [0.0] * (7 - len(params))
        self.connection.mav.command_long_send(
            self.target_system, self.target_component, int(command), 0,
            *[float(value) for value in values[:7]],
        )
        self._record("command_sent", command=int(command), parameters=values[:7])
        return self._wait_ack(command, timeout_s)

    def set_mode(self, mode_name, timeout_s=15.0):
        modes = self.connection.mode_mapping() or {}
        if mode_name not in modes:
            raise ControllerError(f"Mode {mode_name!r} is unavailable; SITL reports: {sorted(modes)}")
        mode = int(modes[mode_name])
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.connection.mav.set_mode_send(
                self.target_system,
                self.mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode,
            )
            self._record("mode_requested", mode=mode_name, custom_mode=mode)
            inner_deadline = time.monotonic() + min(2.0, deadline - time.monotonic())
            while time.monotonic() < inner_deadline:
                heartbeat = self.connection.recv_match(type="HEARTBEAT", blocking=True, timeout=min(0.5, inner_deadline - time.monotonic()))
                if heartbeat is not None and int(getattr(heartbeat, "custom_mode", -1)) == mode:
                    self._record("mode_confirmed", mode=mode_name)
                    return
            time.sleep(0.5)
        raise ControllerError(f"SITL did not confirm mode {mode_name!r} within {timeout_s:g}s")

    def wait_initialization(self, timeout_s: float = 45.0):
        """Wait for ArduPilot sensor calibration, gyro init, EKF3 alignment, and GPS lock."""
        deadline = time.monotonic() + float(timeout_s)
        gps_locked = False
        ekf_ready = False
        prearm_passed = False
        ekf_nav_mask = 1 | 2 | 4 | 8 | 16  # ATTITUDE, VEL_HORIZ, VEL_VERT, POS_HORIZ_REL, POS_HORIZ_ABS
        prearm_flag = int(getattr(self.mavutil.mavlink, "MAV_SYS_STATUS_PREARM_CHECK", 1 << 28))
        start_t = time.monotonic()

        while time.monotonic() < deadline:
            msg = self.connection.recv_match(
                type=["GPS_RAW_INT", "EKF_STATUS_REPORT", "SYS_STATUS", "STATUSTEXT"],
                blocking=True,
                timeout=min(0.5, max(0.05, deadline - time.monotonic())),
            )
            if msg is not None:
                mtype = msg.get_type()
                if mtype == "GPS_RAW_INT":
                    if int(getattr(msg, "fix_type", 0)) >= 3 and int(getattr(msg, "satellites_visible", 0)) >= 6:
                        gps_locked = True
                elif mtype == "EKF_STATUS_REPORT":
                    flags = int(getattr(msg, "flags", 0))
                    if (flags & ekf_nav_mask) == ekf_nav_mask:
                        ekf_ready = True
                elif mtype == "SYS_STATUS":
                    health = int(getattr(msg, "onboard_control_sensors_health", 0))
                    if health & prearm_flag:
                        prearm_passed = True
                elif mtype == "STATUSTEXT":
                    text = str(getattr(msg, "text", ""))
                    if "is using GPS" in text or "Origin set" in text:
                        ekf_ready = True

            elapsed = time.monotonic() - start_t
            if gps_locked and ekf_ready and prearm_passed:
                self._record("initialization_confirmed", gps_locked=True, ekf_ready=True, prearm_passed=True, elapsed_s=round(elapsed, 2))
                return True
            if gps_locked and ekf_ready and elapsed >= 12.0:
                self._record("initialization_confirmed", gps_locked=True, ekf_ready=True, prearm_passed=prearm_passed, elapsed_s=round(elapsed, 2))
                return True

        if gps_locked and ekf_ready:
            self._record("initialization_confirmed", gps_locked=True, ekf_ready=True, prearm_passed=prearm_passed, elapsed_s=round(time.monotonic() - start_t, 2))
            return True

        raise ControllerError(
            f"SITL did not reach initialized / armable state within {timeout_s:g}s "
            f"(gps_locked={gps_locked}, ekf_ready={ekf_ready}, prearm_passed={prearm_passed})"
        )

    def set_cruise_speed(self, speed_mps: float = 16.0):
        """Set horizontal waypoint navigation speed (WP_SPD) and acceptance radius."""
        try:
            self.connection.mav.param_set_send(
                self.target_system,
                self.target_component,
                b"WP_SPD",
                float(speed_mps),
                self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            self.connection.mav.param_set_send(
                self.target_system,
                self.target_component,
                b"WP_RADIUS_M",
                15.0,
                self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            cmd_change_speed = int(getattr(self.mavutil.mavlink, "MAV_CMD_DO_CHANGE_SPEED", 178))
            self._command_long(cmd_change_speed, [1.0, float(speed_mps), -1.0, 0.0, 0.0, 0.0, 0.0], timeout_s=3.0)
        except Exception:
            pass
        self._record("cruise_speed_set", speed_mps=float(speed_mps))

    def arm_guided(self, timeout_s: float):
        # Wait for full sensor, gyro, GPS, and EKF3 initialization before commanding arm
        init_timeout = min(45.0, float(timeout_s))
        self.wait_initialization(timeout_s=init_timeout)

        command_arm = int(self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        deadline = time.monotonic() + timeout_s
        flag = int(self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        modes = self.connection.mode_mapping() or {}
        guided_mode = int(modes.get("GUIDED", 4))
        while time.monotonic() < deadline:
            self.connection.mav.set_mode_send(
                self.target_system,
                self.mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                guided_mode,
            )
            self._record("mode_requested", mode="GUIDED", custom_mode=guided_mode)
            self.connection.mav.command_long_send(
                self.target_system, self.target_component, command_arm, 0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )
            self._record("command_sent", command=command_arm, parameters=[1.0, 0, 0, 0, 0, 0, 0])
            inner_deadline = time.monotonic() + min(2.0, deadline - time.monotonic())
            while time.monotonic() < inner_deadline:
                heartbeat = self.connection.recv_match(type="HEARTBEAT", blocking=True, timeout=min(0.5, inner_deadline - time.monotonic()))
                if heartbeat is not None:
                    is_armed = bool(int(heartbeat.base_mode) & flag)
                    is_guided = int(getattr(heartbeat, "custom_mode", -1)) == guided_mode
                    if is_armed and is_guided:
                        self.live_telemetry.update({
                            "flight_status": "ARMED",
                            "flight_mode": "GUIDED",
                            "armed": True,
                            "last_update_ns": time.monotonic_ns(),
                        })
                        self._record("mode_confirmed", mode="GUIDED")
                        self._record("armed_state", armed=True)
                        return
            time.sleep(1.0)
        raise ControllerError(f"SITL did not reach armed state in GUIDED mode within {timeout_s:g}s")

    def _wait_armed(self, armed: bool, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        flag = int(self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        while time.monotonic() < deadline:
            heartbeat = self.connection.recv_match(type="HEARTBEAT", blocking=True, timeout=min(1.0, deadline - time.monotonic()))
            if heartbeat is not None and bool(int(heartbeat.base_mode) & flag) == bool(armed):
                self.live_telemetry["armed"] = bool(armed)
                self.live_telemetry["last_update_ns"] = time.monotonic_ns()
                self._record("armed_state", armed=bool(armed))
                return
        raise ControllerError(f"SITL did not reach armed={armed} within {timeout_s:g}s")

    def _takeoff_and_wait(self, altitude_m, timeout_s):
        command = int(self.mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
        deadline = time.monotonic() + timeout_s
        target_cm = float(altitude_m) * 100.0 * 0.8
        took_off = False
        self.live_telemetry.update({"flight_status": "TAKING_OFF", "last_update_ns": time.monotonic_ns()})
        while time.monotonic() < deadline:
            if not took_off:
                try:
                    self._command_long(command, [0, 0, 0, 0, 0, 0, float(altitude_m)], timeout_s=3.0)
                    took_off = True
                except Exception:
                    time.sleep(1.0)
                    continue
            message = self.connection.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=min(1.0, deadline - time.monotonic()))
            if message is not None and float(message.relative_alt) / 10.0 >= target_cm:
                altitude = float(message.relative_alt) / 1000.0
                self.live_telemetry.update({
                    "flight_status": "AIRBORNE",
                    "relative_alt_m": round(altitude, 2),
                    "last_update_ns": time.monotonic_ns(),
                })
                self._record("takeoff_confirmed", relative_altitude_m=altitude)
                return altitude
        raise ControllerError(f"SITL did not reach {target_cm / 100:g}m takeoff altitude within {timeout_s:g}s")

    def _send_position_target(self, target_lat: int, target_lon: int, target_alt_m: float):
        frame_rel = int(self.mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT)
        # Type mask: 3576 (0b0000110111111000): ignore velocity, acceleration, yaw, yaw_rate
        self.connection.mav.set_position_target_global_int_send(
            0,
            self.target_system,
            self.target_component,
            frame_rel,
            3576,
            int(target_lat),
            int(target_lon),
            float(target_alt_m),
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
        )
        # Attempt command acknowledgment via MAV_CMD_DO_REPOSITION if supported
        try:
            cmd_reposition = int(getattr(self.mavutil.mavlink, "MAV_CMD_DO_REPOSITION", 192))
            self.connection.mav.command_int_send(
                self.target_system, self.target_component,
                frame_rel, cmd_reposition,
                0, 0,
                -1.0, 1.0, 0.0, float("nan"),
                int(target_lat), int(target_lon), float(target_alt_m),
            )
            self._record("command_sent", command=cmd_reposition, target_lat=int(target_lat), target_lon=int(target_lon), target_alt_m=float(target_alt_m))
            self._wait_ack(cmd_reposition, timeout_s=1.5)
        except Exception:
            pass

    def _fly_to_waypoint(self, target_lat: int, target_lon: int, target_alt_m: float,
                         acceptance_radius_m: float = 25.0, timeout_s: float = 90.0,
                         cos_lat: float = 1.0, lap: int = 1, wp_index: int = 1):
        deadline = time.monotonic() + timeout_s
        self._record("waypoint_commanded", lap=lap, wp_index=wp_index, target_lat=int(target_lat), target_lon=int(target_lon), target_alt_m=float(target_alt_m))
        self._send_position_target(target_lat, target_lon, target_alt_m)
        last_sent = time.monotonic()
        dist_m = 999.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_sent >= 1.5:
                self._send_position_target(target_lat, target_lon, target_alt_m)
                last_sent = now
            msg = self.connection.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=min(0.5, max(0.05, deadline - now)))
            if msg is not None:
                cur_lat = int(msg.lat)
                cur_lon = int(msg.lon)
                cur_alt_m = float(msg.relative_alt) / 1000.0
                vx = float(getattr(msg, "vx", 0)) / 100.0
                vy = float(getattr(msg, "vy", 0)) / 100.0
                hdg = float(getattr(msg, "hdg", 0)) / 100.0
                dy_m = ((target_lat - cur_lat) / 1e7) * 111139.0
                dx_m = ((target_lon - cur_lon) / 1e7) * 111139.0 * cos_lat
                dist_m = math.hypot(dx_m, dy_m)
                self.live_telemetry.update({
                    "flight_status": "FLYING_CIRCUIT",
                    "flight_mode": "GUIDED",
                    "armed": True,
                    "relative_alt_m": round(cur_alt_m, 2),
                    "lat": cur_lat,
                    "lon": cur_lon,
                    "groundspeed_mps": round(math.hypot(vx, vy), 2),
                    "heading_deg": round(hdg, 1),
                    "lap": lap,
                    "waypoint": wp_index,
                    "distance_to_wp_m": round(dist_m, 1),
                    "last_update_ns": time.monotonic_ns(),
                })
                if dist_m <= acceptance_radius_m:
                    self._record(
                        "waypoint_confirmed",
                        lap=lap,
                        wp_index=wp_index,
                        target_lat=int(target_lat),
                        target_lon=int(target_lon),
                        actual_lat=cur_lat,
                        actual_lon=cur_lon,
                        relative_alt_m=round(cur_alt_m, 2),
                        distance_m=round(dist_m, 2),
                    )
                    return dist_m
        raise ControllerError(
            f"Drone timed out reaching waypoint {wp_index} in lap {lap} within {timeout_s:g}s "
            f"(target: ({target_lat/1e7:.6f}, {target_lon/1e7:.6f}), current dist: {dist_m:.1f}m)"
        )

    def fly_route_mission(self, laps: int = 2, north_m: float = 180.0, east_m: float = 120.0,
                          altitude_m: float = 15.0, cruise_speed_mps: float = 16.0,
                          acceptance_radius_m: float = 15.0, pause_at_wp_s: float = 0.5,
                          post_landing_hold_s: float = 15.0, radius_m: float = None,
                          settings: dict = None, on_airborne = None, on_landed = None,
                          min_flight_duration_s: float = None, on_waypoint_event = None,
                          on_circuit_completed = None):
        """Take off, fly a balanced multi-lap circuit around the airfield, return to home, and land."""
        if laps < 1:
            raise ValueError("Route laps must be at least 1")
        settings = settings or {}
        arm_timeout = float(settings.get("arm_timeout_s", 60.0))
        takeoff_timeout = float(settings.get("takeoff_timeout_s", 45.0))
        land_timeout = float(settings.get("land_timeout_s", 60.0))

        # Handle backwards-compatible radius_m if explicitly specified
        if radius_m is not None and radius_m != 12.0 and north_m == 180.0:
            scale = float(radius_m) / 12.0
            north_m = 180.0 * scale
            east_m = 120.0 * scale

        self.wait_heartbeat(arm_timeout)
        self.set_cruise_speed(cruise_speed_mps)
        self.arm_guided(arm_timeout)
        actual_takeoff_alt = self._takeoff_and_wait(altitude_m, takeoff_timeout)

        # Get initial position as home
        msg = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and msg is None:
            msg = self.connection.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if msg is None:
            raise ControllerError("Failed to obtain GLOBAL_POSITION_INT after takeoff for route positioning")

        home_lat = int(msg.lat)
        home_lon = int(msg.lon)
        cos_lat = math.cos(math.radians(home_lat / 1e7))

        d_lat_full = int((north_m / 111139.0) * 1e7)
        d_lat_half = int((north_m * 0.5 / 111139.0) * 1e7)
        d_lon_full = int((east_m / (111139.0 * max(0.1, cos_lat))) * 1e7)
        d_lon_half = int((east_m * 0.5 / (111139.0 * max(0.1, cos_lat))) * 1e7)

        # 8-waypoint broad oval circuit centered on Canberra airport launch point:
        # WP1 (NE1), WP2 (NE2), WP3 (SE1), WP4 (SE2), WP5 (SW1), WP6 (SW2), WP7 (NW1), WP8 (NW2)
        waypoints = [
            (home_lat + d_lat_full, home_lon + d_lon_half),  # WP1: North-East 1
            (home_lat + d_lat_half, home_lon + d_lon_full),  # WP2: North-East 2
            (home_lat - d_lat_half, home_lon + d_lon_full),  # WP3: South-East 1 (East straight)
            (home_lat - d_lat_full, home_lon + d_lon_half),  # WP4: South-East 2
            (home_lat - d_lat_full, home_lon - d_lon_half),  # WP5: South-West 1 (South straight)
            (home_lat - d_lat_half, home_lon - d_lon_full),  # WP6: South-West 2
            (home_lat + d_lat_half, home_lon - d_lon_full),  # WP7: North-West 1 (West straight)
            (home_lat + d_lat_full, home_lon - d_lon_half),  # WP8: North-West 2 (North straight)
        ]

        # Calculate exact geometric circuit distances
        corner_dist = math.hypot(north_m * 0.5, east_m * 0.5)
        ns_straight = north_m
        ew_straight = east_m
        lap_distance_m = round(2.0 * ns_straight + 2.0 * ew_straight + 4.0 * corner_dist, 1)
        total_distance_m = round(lap_distance_m * laps, 1)

        flight_start_ns = time.monotonic_ns()
        self._record(
            "mission_airborne",
            flight_start_monotonic_ns=flight_start_ns,
            route=True,
            laps=laps,
            north_m=north_m,
            east_m=east_m,
            lap_distance_m=lap_distance_m,
            total_distance_m=total_distance_m,
            cruise_speed_mps=cruise_speed_mps,
            altitude_m=float(altitude_m),
            home_lat=home_lat,
            home_lon=home_lon,
        )
        if on_airborne is not None:
            on_airborne(flight_start_ns)

        waypoints_completed = 0
        lap = 1
        target_duration = float(min_flight_duration_s) if min_flight_duration_s is not None else 0.0
        while True:
            self._record("lap_started", lap=lap, total_laps=laps)
            break_circuit = False
            for wp_idx, (wp_lat, wp_lon) in enumerate(waypoints, start=1):
                self._fly_to_waypoint(
                    wp_lat, wp_lon, altitude_m,
                    acceptance_radius_m=acceptance_radius_m, timeout_s=90.0,
                    cos_lat=cos_lat, lap=lap, wp_index=wp_idx,
                )
                waypoints_completed += 1
                if callable(on_waypoint_event):
                    try:
                        on_waypoint_event(lap, wp_idx, waypoints_completed)
                    except Exception:
                        pass
                if pause_at_wp_s > 0:
                    time.sleep(pause_at_wp_s)
                elapsed_s = (time.monotonic_ns() - flight_start_ns) / 1e9
                if min_flight_duration_s is not None and elapsed_s >= target_duration and (lap >= laps or waypoints_completed >= laps * len(waypoints)):
                    break_circuit = True
                    break
            self._record("lap_completed", lap=lap, total_laps=laps)
            if break_circuit:
                break
            elapsed_s = (time.monotonic_ns() - flight_start_ns) / 1e9
            if min_flight_duration_s is not None:
                if elapsed_s >= target_duration and (lap >= laps or waypoints_completed >= laps * len(waypoints)):
                    break
            else:
                if lap >= laps:
                    break
            lap += 1

        # Circuit complete: trigger callback before returning home and landing
        if callable(on_circuit_completed):
            try:
                on_circuit_completed()
            except Exception:
                pass
        self._record("circuit_completed", laps_completed=lap, total_waypoints=waypoints_completed)

        # Return to home before landing
        self.live_telemetry.update({"flight_status": "RETURNING_HOME", "last_update_ns": time.monotonic_ns()})
        self._record("return_to_home_started", home_lat=home_lat, home_lon=home_lon)
        self._fly_to_waypoint(
            home_lat, home_lon, altitude_m,
            acceptance_radius_m=acceptance_radius_m, timeout_s=90.0,
            cos_lat=cos_lat, lap=lap, wp_index=0,
        )
        self._record("return_to_home_confirmed")
        if pause_at_wp_s > 0:
            time.sleep(pause_at_wp_s)

        # Land and wait for disarm
        self.live_telemetry.update({"flight_status": "LANDING", "flight_mode": "LAND", "last_update_ns": time.monotonic_ns()})
        self.set_mode("LAND", timeout_s=15.0)
        self._wait_armed(False, land_timeout)
        self.live_telemetry.update({"flight_status": "LANDED", "armed": False, "last_update_ns": time.monotonic_ns()})
        flight_end_ns = time.monotonic_ns()
        total_duration_s = (flight_end_ns - flight_start_ns) / 1e9
        actual_total_distance_m = round(lap_distance_m * (waypoints_completed / len(waypoints)), 1)
        self._record(
            "mission_landed",
            flight_duration_s=round(total_duration_s, 2),
            waypoints_completed=waypoints_completed,
            lap_distance_m=lap_distance_m,
            total_distance_m=actual_total_distance_m,
        )

        # Post-landing hold: keep telemetry alive so QGroundControl displays the landed state and full trail without "Comms Lost"
        if post_landing_hold_s > 0:
            self._record("post_landing_hold_started", hold_s=float(post_landing_hold_s))
            if callable(on_landed):
                try:
                    on_landed()
                except Exception:
                    pass
            time.sleep(float(post_landing_hold_s))
            self._record("post_landing_hold_completed")

        return {
            "flight_start_monotonic_ns": flight_start_ns,
            "flight_duration_s": round(total_duration_s, 2),
            "route": True,
            "laps": lap,
            "north_m": north_m,
            "east_m": east_m,
            "lap_distance_m": lap_distance_m,
            "total_distance_m": actual_total_distance_m,
            "cruise_speed_mps": cruise_speed_mps,
            "altitude_m": float(altitude_m),
            "pause_at_wp_s": pause_at_wp_s,
            "waypoints_completed": waypoints_completed,
            "returned_to_home": True,
            "landed": True,
        }

    def run_benign_mission(self, dwell_s, altitude_m, settings, on_airborne=None,
                           route=True, laps=2, north_m=180.0, east_m=120.0,
                           cruise_speed_mps=16.0, acceptance_radius_m=15.0,
                           pause_at_wp_s=0.5, post_landing_hold_s=15.0, radius_m=None,
                           on_landed=None, min_flight_duration_s=None, on_waypoint_event=None,
                           on_circuit_completed=None):
        """Take off, fly either multi-lap circuit route (default for benign) or fixed dwell hover, and land."""
        if route:
            return self.fly_route_mission(
                laps=int(laps),
                north_m=float(north_m),
                east_m=float(east_m),
                altitude_m=float(altitude_m),
                cruise_speed_mps=float(cruise_speed_mps),
                acceptance_radius_m=float(acceptance_radius_m),
                pause_at_wp_s=float(pause_at_wp_s),
                post_landing_hold_s=float(post_landing_hold_s),
                radius_m=radius_m,
                settings=settings,
                on_airborne=on_airborne,
                on_landed=on_landed,
                min_flight_duration_s=min_flight_duration_s,
                on_waypoint_event=on_waypoint_event,
                on_circuit_completed=on_circuit_completed,
            )
        if float(dwell_s) < 1:
            raise ValueError("Mission dwell must be at least one second")
        self.wait_heartbeat(settings["arm_timeout_s"])
        self.arm_guided(float(settings["arm_timeout_s"]))
        self._takeoff_and_wait(altitude_m, float(settings["takeoff_timeout_s"]))
        flight_start_ns = time.monotonic_ns()
        self._record("mission_airborne", flight_start_monotonic_ns=flight_start_ns, dwell_s=float(dwell_s))
        if on_airborne is not None:
            on_airborne(flight_start_ns)
        time.sleep(float(dwell_s))
        self.set_mode("LAND", timeout_s=15.0)
        self._wait_armed(False, float(settings["land_timeout_s"]))
        self._record("mission_landed")
        if post_landing_hold_s > 0:
            time.sleep(float(post_landing_hold_s))
        return {"flight_start_monotonic_ns": flight_start_ns, "dwell_s": float(dwell_s), "altitude_m": float(altitude_m)}

    def close(self):
        try:
            self.connection.close()
        except Exception:
            pass
