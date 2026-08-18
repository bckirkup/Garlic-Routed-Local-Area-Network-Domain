"""Re-pick the zone trigger count from the fleet's own quiet token rate.

``privacy.threshold_m`` is the count of genuine anomaly tokens a zone x anomaly
pair needs inside the aggregation window before the aggregator broadcasts. It
was chosen against the token volume the *uncalibrated* detector produced. Once
``garland.alarm_calibration`` holds the quiet-epoch alarm rate at the rate the
configured cut implies, every agent emits fewer tokens, so the same fixed count
became a stricter test than it was when it was picked: on the 2K town scenario,
enabling alarm calibration cut zone-level warranted detections from 7 to 1
while the per-epoch detector false-positive rate barely moved.

Fixing that by lowering the constant would only be right for one population
density. The count that matters is a *rate* question: how often does a quiet
zone-window accumulate m tokens by chance? This module answers it the same way
the alarm calibrator does -- accumulate the window counts the aggregator
actually sees over a quiet calibration window, in a histogram, then read off
the smallest count whose empirical upper tail is at or below a configured
per-evaluation false-trigger rate, and freeze it.

Properties worth knowing:

* It is floored at ``minimum_threshold`` (2 by default), so no calibration
  outcome lets a single anomalous device broadcast its own zone.
* It only calibrates from windows the aggregator evaluated during the quiet
  window, so a hazard inside that window inflates the learned count and costs
  sensitivity later -- the same failure mode the alarm calibrator has.
* It is a single fleet-level count, not one per zone. Zones differ in device
  occupancy, so a pooled count is stricter than necessary for sparse zones and
  looser than necessary for dense ones; the pooled tail is nonetheless measured
  against the density the run actually has, which the constant never was.

See ``docs/OPERATIONAL_DETECTION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Window counts above this are treated as one bucket: they clear any threshold
# this module is allowed to pick, so they only need to be counted.
MAX_TRACKED_COUNT = 64


@dataclass
class ZoneThresholdCalibrationConfig:
    """Settings for the fleet zone-trigger recalibration.

    Parameters
    ----------
    enabled : bool
        Whether the aggregator re-derives its trigger count from measured
        quiet-window token counts. Off by default, so a run reproduces the
        configured ``privacy.threshold_m`` unless recalibration is requested.
    start_step : int
        First step whose zone-window evaluations enter the histogram. Later
        than zero because agent baselines are still cold, and because the
        detector's own alarm scale has not frozen yet.
    end_step : int
        Step at which the trigger count is computed and frozen.
    false_trigger_rate : float
        Target probability that one quiet zone x anomaly evaluation reaches the
        trigger count. This is the operating point the count is chosen for.
    minimum_threshold : int
        Floor on the learned count. Two tokens means a broadcast always rests
        on at least two devices.
    maximum_threshold : int
        Ceiling on the learned count, so an unexpectedly noisy calibration
        window cannot switch zone detection off altogether.
    min_samples : int
        Zone-window evaluations required before the histogram is trusted;
        below that the configured ``threshold_m`` stands.
    """

    enabled: bool = False
    start_step: int = 144
    end_step: int = 720
    false_trigger_rate: float = 0.01
    minimum_threshold: int = 2
    maximum_threshold: int = 32
    min_samples: int = 500

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if self.end_step <= self.start_step:
            raise ValueError(
                f"end_step must exceed start_step, got {self.end_step} <= {self.start_step}"
            )
        if not 0.0 < self.false_trigger_rate < 1.0:
            raise ValueError(
                f"false_trigger_rate must lie strictly in (0, 1), got {self.false_trigger_rate}"
            )
        if self.minimum_threshold < 2:
            raise ValueError(f"minimum_threshold must be at least 2, got {self.minimum_threshold}")
        if self.maximum_threshold < self.minimum_threshold:
            raise ValueError(
                "maximum_threshold must be at least minimum_threshold, got "
                f"{self.maximum_threshold} < {self.minimum_threshold}"
            )
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {self.min_samples}")


@dataclass
class ZoneThresholdCalibrator:
    """Fleet-level replacement for a hand-picked ``threshold_m``."""

    config: ZoneThresholdCalibrationConfig = field(default_factory=ZoneThresholdCalibrationConfig)
    configured_threshold: int = 5
    collecting: bool = False
    frozen: bool = False
    threshold: int = 5
    _counts: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(MAX_TRACKED_COUNT + 1, dtype=np.int64)
    )

    def __post_init__(self) -> None:
        if self.configured_threshold < 1:
            raise ValueError(
                f"configured_threshold must be positive, got {self.configured_threshold}"
            )
        self.threshold = self.configured_threshold

    @property
    def samples(self) -> int:
        """Zone-window evaluations folded into the histogram so far."""
        return int(self._counts.sum())

    def observe(self, window_count: int) -> None:
        """Fold one zone x anomaly window count into the histogram."""
        if not self.collecting or self.frozen or window_count < 0:
            return
        self._counts[min(window_count, MAX_TRACKED_COUNT)] += 1

    def _learned_threshold(self) -> int | None:
        """Smallest count whose measured upper tail meets the target rate."""
        total = self.samples
        if total < self.config.min_samples:
            return None
        # Upper tail at count c is the share of evaluations reaching c or more.
        at_or_above = np.cumsum(self._counts[::-1])[::-1] / total
        for count in range(self.config.minimum_threshold, self.config.maximum_threshold + 1):
            if count > MAX_TRACKED_COUNT:
                break
            if at_or_above[count] <= self.config.false_trigger_rate:
                return count
        return self.config.maximum_threshold

    def freeze(self) -> None:
        """Compute the trigger count from the histogram and stop collecting."""
        self.collecting = False
        if self.frozen:
            return
        self.frozen = True
        learned = self._learned_threshold()
        if learned is not None:
            self.threshold = learned

    def advance(self, step: int) -> None:
        """Open, hold or close the calibration window for simulation ``step``."""
        if self.frozen:
            return
        if step >= self.config.end_step:
            self.freeze()
            return
        self.collecting = step >= self.config.start_step

    def summary(self) -> dict[str, object]:
        """Learned count and how much evidence it rests on."""
        return {
            "frozen": self.frozen,
            "configured_threshold_m": self.configured_threshold,
            "threshold_m": self.threshold,
            "calibration_windows": self.samples,
            "target_false_trigger_rate": self.config.false_trigger_rate,
        }


def zone_threshold_calibrator_for(
    config: ZoneThresholdCalibrationConfig,
    configured_threshold: int,
) -> ZoneThresholdCalibrator:
    """Build a calibrator that starts from the configured trigger count."""
    return ZoneThresholdCalibrator(config=config, configured_threshold=configured_threshold)


__all__ = [
    "MAX_TRACKED_COUNT",
    "ZoneThresholdCalibrationConfig",
    "ZoneThresholdCalibrator",
    "zone_threshold_calibrator_for",
]
