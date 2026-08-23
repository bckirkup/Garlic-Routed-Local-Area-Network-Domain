"""Hold the fleet's quiet-epoch alarm rate at the rate the cut implies.

``garland.thresholds`` re-expresses a configured cut at whatever width an agent
reports, which keeps the alarm rate stable *if* the scored residuals are jointly
Gaussian. Much of the wide fleet is not: step counts, cough rate, wheeze and
crackle burden, bladder impedance and ectopy burden are floored at zero and
right-skewed, so the null distribution of the Mahalanobis statistic has a far
heavier right tail than chi-square. Measured on a hazard-free, confounder-free
run with every subsystem adopted, the chi-square cut that should flag 1.6% of
quiet epochs flagged 4.5% of them at 6-12 channels and 9.4% at 21-30 -- the
width-dependent false-positive rise that ``detection_power`` was built to
expose, and the reason a wider vector looked more sensitive than it was.

This module corrects it empirically instead of assuming a distribution. Over a
calibration window the fleet accumulates the ratio of each scored distance to
the chi-square cut for its width, in a histogram rather than a sample list, and
then reads off the upper quantile that corresponds to the target alarm rate.
That quantile becomes a multiplicative scale on the cut, per width bucket, and
is then frozen for the rest of the run.

Two properties matter for what the scale can and cannot do:

* It is floored at 1.0, so calibration only ever makes agents *less*
  trigger-happy than the configured cut. A four-channel run, whose empirical
  rate already sits near target, is left essentially untouched.
* It is fleet-level and frozen, not per-agent and adaptive. An agent that keeps
  adapting to its own alarms would tune away a real sustained hazard signal --
  measured on the 2K town scenario, a per-agent Robbins-Monro variant of this
  correction halved the false-positive rate but also lost the outbreak. A cut
  calibrated once against a mostly quiet reference population, which is how a
  device vendor would ship one, cannot desensitize itself during an episode.

The window should therefore sit in quiet time: hazard-affected epochs inside it
inflate the learned scale, which costs sensitivity later.

See ``docs/OPERATIONAL_DETECTION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.detection_power import WIDTH_BUCKET_LABELS, width_bucket
from garland.thresholds import REFERENCE_DOF, per_epoch_false_positive_rate, threshold_for_dof

# Resolution of the ratio histogram the quantile is read from. Ratios above the
# top bin are alarms under any scale this module will accept, so they only need
# to be counted, not resolved.
RATIO_BIN_WIDTH = 0.01
RATIO_BINS = 1000


@dataclass
class AlarmCalibrationConfig:
    """Settings for the fleet alarm-rate calibration.

    Parameters
    ----------
    enabled : bool
        Whether the fleet scales its cut to the configured alarm rate.
    start_step : int
        First step whose scored epochs enter the calibration histogram. Later
        than zero because a cold baseline has not learned its own residual
        covariance yet.
    end_step : int
        Step at which the scales are computed and frozen.
    max_scale : float
        Ceiling on the scale, so one pathological channel cannot silence a
        width bucket outright.
    min_samples : int
        Scored epochs a width bucket needs before it is calibrated on its own
        ratios; below that it borrows its nearest well-sampled neighbour's
        scale, because the inflation grows with width and a neighbouring width
        is a far better guess than the fleet-pooled distribution (which is
        dominated by whichever width most people happen to wear).
    defer_broadcasts_until_frozen : bool
        Whether the aggregation layer withholds broadcasts while the scales are
        still being measured. Tokens minted before the freeze come from cuts the
        run has already established are miscalibrated, and on the 2K town they
        outnumber every later token: at ``wearable_fraction`` 0.6 the 720 steps
        before the freeze produced 9,011 of the run's 9,873 zone triggers and
        none of its warranted detections, each one spending response epsilon.
        Off by default, because it makes the aggregation layer silent for any
        run shorter than ``end_step`` and blind to a hazard that both starts and
        ends inside the calibration window; a scenario long enough to reach the
        freeze, and whose hazards arrive after it, should turn it on.
    """

    enabled: bool = True
    start_step: int = 144
    end_step: int = 720
    max_scale: float = 3.0
    min_samples: int = 500
    defer_broadcasts_until_frozen: bool = False

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if self.end_step <= self.start_step:
            raise ValueError(
                f"end_step must exceed start_step, got {self.end_step} <= {self.start_step}"
            )
        if self.max_scale < 1.0:
            raise ValueError(f"max_scale must be at least 1, got {self.max_scale}")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {self.min_samples}")


def _quantile_from_histogram(counts: NDArray[np.int64], tail_probability: float) -> float | None:
    """Ratio whose upper tail in ``counts`` is ``tail_probability``.

    The final bin collects everything above the histogram's range; when the
    target quantile falls inside it the ratio is not resolvable and the caller
    has to fall back rather than invent a value.
    """
    total = int(counts.sum())
    if total == 0:
        return None
    target = total * (1.0 - tail_probability)
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, target, "left"))
    if index >= RATIO_BINS:
        return None
    below = int(cumulative[index - 1]) if index else 0
    in_bin = int(counts[index])
    # Linear interpolation inside the bin keeps the scale continuous as the
    # histogram fills, rather than stepping by a whole bin width.
    offset = 0.0 if in_bin == 0 else (target - below) / in_bin
    return float((index + offset) * RATIO_BIN_WIDTH)


def _nearest_fitted_position(fitted: dict[int, float], position: int) -> int:
    """Return the fitted bucket position closest to ``position``, ties toward lower."""
    return min(fitted, key=lambda other: (abs(other - position), other))


@dataclass
class AlarmRateCalibrator:
    """Fleet-level multiplicative correction to dof-calibrated cuts."""

    config: AlarmCalibrationConfig = field(default_factory=AlarmCalibrationConfig)
    target_rate: float = per_epoch_false_positive_rate(3.5, REFERENCE_DOF)
    collecting: bool = False
    frozen: bool = False
    scales: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(WIDTH_BUCKET_LABELS, 1.0)
    )
    _counts: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros((len(WIDTH_BUCKET_LABELS), RATIO_BINS + 1), dtype=np.int64)
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.target_rate < 1.0:
            raise ValueError(f"target_rate must lie strictly in (0, 1), got {self.target_rate}")

    @property
    def samples(self) -> int:
        """Scored epochs folded into the calibration histogram so far."""
        return int(self._counts.sum())

    def scale_for(self, width: int) -> float:
        """Return the scale a ``width``-wide epoch should be scored against."""
        return self.scales[width_bucket(width)]

    def threshold(self, reference_threshold: float, dof: int) -> float:
        """Return the calibrated cut for a ``dof``-wide epoch."""
        return threshold_for_dof(reference_threshold, dof) * self.scale_for(dof)

    def observe(self, width: int, ratio: float) -> None:
        """Fold one scored epoch's distance-to-cut ratio into the histogram."""
        if not self.collecting or self.frozen or width < 1:
            return
        if not np.isfinite(ratio):
            return
        bucket = WIDTH_BUCKET_LABELS.index(width_bucket(width))
        bin_index = min(int(max(ratio, 0.0) / RATIO_BIN_WIDTH), RATIO_BINS)
        self._counts[bucket, bin_index] += 1

    def _clamped(self, quantile: float) -> float:
        """Keep a learned quantile inside the configured scale bounds."""
        return min(max(quantile, 1.0), self.config.max_scale)

    def freeze(self) -> None:
        """Compute the per-bucket scales from the histogram and stop collecting."""
        self.collecting = False
        if self.frozen:
            return
        self.frozen = True
        fitted: dict[int, float] = {}
        for position in range(len(WIDTH_BUCKET_LABELS)):
            counts = self._counts[position]
            if int(counts.sum()) < self.config.min_samples:
                continue
            quantile = _quantile_from_histogram(counts, self.target_rate)
            if quantile is not None:
                fitted[position] = self._clamped(quantile)
        pooled = _quantile_from_histogram(self._counts.sum(axis=0), self.target_rate)
        for position, label in enumerate(WIDTH_BUCKET_LABELS):
            if position in fitted:
                self.scales[label] = fitted[position]
            elif fitted:
                nearest = _nearest_fitted_position(fitted, position)
                self.scales[label] = fitted[nearest]
            elif pooled is not None:
                self.scales[label] = self._clamped(pooled)

    def advance(self, step: int) -> None:
        """Open, hold or close the calibration window for simulation ``step``."""
        if self.frozen:
            return
        if step >= self.config.end_step:
            self.freeze()
            return
        self.collecting = step >= self.config.start_step

    def summary(self) -> dict[str, object]:
        """Learned scales and how much evidence they rest on."""
        return {
            "frozen": self.frozen,
            "calibration_epochs": self.samples,
            "scales": dict(self.scales),
        }


def calibrator_for(
    config: AlarmCalibrationConfig,
    reference_threshold: float,
) -> AlarmRateCalibrator:
    """Build a calibrator whose target is the rate ``reference_threshold`` implies."""
    return AlarmRateCalibrator(
        config=config,
        target_rate=per_epoch_false_positive_rate(reference_threshold, REFERENCE_DOF),
    )


__all__ = [
    "RATIO_BINS",
    "RATIO_BIN_WIDTH",
    "AlarmCalibrationConfig",
    "AlarmRateCalibrator",
    "calibrator_for",
]
