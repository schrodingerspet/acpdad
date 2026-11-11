"""Causal telemetry silence watchdog for MAVLink position updates.

Monitors the arrival stream of GLOBAL_POSITION_INT messages strictly forward in time.
Learns nominal packet inter-arrival bounds from benign training and validation data.
When telemetry updates stop or become excessively delayed, the watchdog raises an
operational DoS alert during the silent period.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


class CausalTelemetryWatchdog:
    """Schedule-independent, causal monitor for missing or late GLOBAL_POSITION_INT telemetry.

    Parameters
    ----------
    silence_threshold_s : float
        Inter-arrival threshold in seconds beyond which telemetry is deemed silent.
        Learned from benign SITL telemetry (mean 0.250s, max 0.295s; default 0.450s).
    consecutive_timely_required : int
        Number of consecutive timely packets required to clear the silence alarm (default 2).
    """

    def __init__(
        self,
        silence_threshold_s: float = 0.450,
        consecutive_timely_required: int = 2,
    ) -> None:
        self.silence_threshold_s = float(silence_threshold_s)
        self.consecutive_timely_required = int(consecutive_timely_required)
        self.last_arrival_ns: Optional[int] = None
        self.timely_count: int = 0
        self.is_silent: bool = False
        self.silence_alert_count: int = 0

    def reset(self) -> None:
        """Reset internal state at run initialization."""
        self.last_arrival_ns = None
        self.timely_count = 0
        self.is_silent = False
        self.silence_alert_count = 0

    def update(self, arrival_time_ns: int) -> Dict[str, Any]:
        """Causally update state on packet arrival.

        Returns status dict including whether the packet arrived during an active silence alert.
        """
        arrival_time_ns = int(arrival_time_ns)
        was_silent = self.is_silent
        delta_s = (
            (arrival_time_ns - self.last_arrival_ns) / 1e9
            if self.last_arrival_ns is not None
            else 0.0
        )

        if self.last_arrival_ns is not None and delta_s > self.silence_threshold_s:
            # Excessive gap between delivered packets: entered or continued silence
            self.is_silent = True
            self.timely_count = 0
            self.silence_alert_count += 1
        elif self.last_arrival_ns is not None:
            # Timely arrival
            self.timely_count += 1
            if self.timely_count >= self.consecutive_timely_required:
                self.is_silent = False

        self.last_arrival_ns = arrival_time_ns

        return {
            "arrival_time_ns": arrival_time_ns,
            "delta_s": delta_s,
            "is_silent": self.is_silent,
            "was_silent": was_silent,
            "operational_alert": "DoS" if self.is_silent else "Benign",
        }

    def check_at_time(self, query_time_ns: int) -> bool:
        """Causally check if the monitor is in a silent state at query_time_ns without lookahead."""
        if self.last_arrival_ns is None:
            return False
        gap_s = (int(query_time_ns) - self.last_arrival_ns) / 1e9
        return gap_s > self.silence_threshold_s
