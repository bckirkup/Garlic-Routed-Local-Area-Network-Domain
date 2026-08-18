"""Detection-power instrumentation stratified by observation width and device.

The rest of the metrics layer measures the *system*: how long a zone took to
alarm, how often a quiet zone did. Neither answers the question a mixed-modality
fleet raises, which is whether adopting a sensor subsystem buys any detection
power at all. That needs outcomes keyed by what a person was actually wearing
when they were scored:

- **Effective width** — how many channels were present *and* unmasked this
  epoch. Structural missingness (a subsystem nobody adopted) and duty-cycle
  masking (a subsystem that reported nothing this epoch) both reduce it, so the
  same person moves between width buckets over a day.
- **Per-epoch true- and false-positive rates by width.** Under the calibration
  in ``garland.thresholds`` the false-positive rate is meant to be flat in
  width; a rate that climbs with width falsifies that calibration, and a
  true-positive rate that does *not* climb with width falsifies the premise that
  extra channels are worth wearing.
- **Detection latency by width**, counted from the epoch an agent first became
  hazard-affected to the first epoch it emitted a token.
- **Per-device telemetry** — how much of each subsystem's channel budget
  survived duty cycling, and the outcome rates among its owners.
- **Drop-one-channel ablation** — an optional diagnostic that re-scores alarming
  epochs with each channel removed in turn. Channels here are deliberately weak
  or non-diagnostic, so detection is supposed to be *collective*: one channel
  whose removal cancels most alarms would mean the fleet is really a
  single-channel detector wearing a costume.

Rates are per scored epoch and per agent, not per zone, so they measure the
sensing layer before K-anonymity dilution and aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.biometrics import BaselineTracker
from garland.channels import ChannelSet
from garland.devices import DeviceFleet
from garland.thresholds import threshold_for_dof

# Lower bound of each effective-width bucket, ascending. The first bucket is the
# historical four core vitals plus a spare; the rest roughly double, so a person
# wearing one extra band and one wearing the whole fleet do not share a label.
WIDTH_BUCKET_LOWER_BOUNDS: tuple[int, ...] = (1, 6, 13, 25)
WIDTH_BUCKET_LABELS: tuple[str, ...] = ("1-5", "6-12", "13-24", "25+")


def width_bucket(width: int) -> str:
    """Label the effective-width bucket ``width`` falls in."""
    if width < 1:
        raise ValueError("width must be at least 1 to fall in a bucket")
    return WIDTH_BUCKET_LABELS[int(np.searchsorted(WIDTH_BUCKET_LOWER_BOUNDS, width, "right")) - 1]


def _bucket_ids(widths: NDArray[np.int_]) -> NDArray[np.int_]:
    """Bucket index per width; widths below one are not in any bucket."""
    return np.searchsorted(WIDTH_BUCKET_LOWER_BOUNDS, widths, "right") - 1


@dataclass
class DetectionPowerConfig:
    """Configuration for the detection-power instrumentation.

    The width and device telemetry is vectorized over the fleet and always on.
    ``channel_ablation_rate`` gates the drop-one-channel diagnostic, which costs
    one extra Mahalanobis evaluation per observed channel on each sampled
    alarming epoch and so defaults to off.
    """

    channel_ablation_rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.channel_ablation_rate <= 1.0:
            raise ValueError(
                f"channel_ablation_rate must be in [0, 1], got {self.channel_ablation_rate}"
            )


@dataclass
class _WidthCell:
    """Scored-epoch outcomes for one effective-width bucket."""

    epochs: int = 0
    width_sum: int = 0
    tokens: int = 0
    hazard_epochs: int = 0
    hazard_tokens: int = 0
    clean_epochs: int = 0
    clean_tokens: int = 0
    latencies: int = 0
    latency_sum: int = 0
    latency_min: int | None = None
    latency_max: int | None = None

    def record_latency(self, latency: int) -> None:
        """Fold one first-detection latency into the running summary."""
        self.latencies += 1
        self.latency_sum += latency
        self.latency_min = latency if self.latency_min is None else min(self.latency_min, latency)
        self.latency_max = latency if self.latency_max is None else max(self.latency_max, latency)

    def summary(self) -> dict[str, float | int | None]:
        """Rates and counts for this bucket; rates are None when unobserved."""
        return {
            "scored_epochs": self.epochs,
            "mean_effective_width": _ratio(self.width_sum, self.epochs),
            "tokens": self.tokens,
            "token_rate": _ratio(self.tokens, self.epochs),
            "hazard_epochs": self.hazard_epochs,
            "true_positive_rate": _ratio(self.hazard_tokens, self.hazard_epochs),
            "clean_epochs": self.clean_epochs,
            "false_positive_rate": _ratio(self.clean_tokens, self.clean_epochs),
            "detections_timed": self.latencies,
            "mean_detection_latency_steps": _ratio(self.latency_sum, self.latencies),
            "min_detection_latency_steps": self.latency_min,
            "max_detection_latency_steps": self.latency_max,
        }


@dataclass
class _DeviceCell:
    """Reporting yield and outcomes among the owners of one device kind."""

    owners: int = 0
    owner_epochs: int = 0
    channel_slots: int = 0
    observed_slots: int = 0
    reporting_epochs: int = 0
    tokens: int = 0
    hazard_epochs: int = 0
    hazard_tokens: int = 0
    clean_epochs: int = 0
    clean_tokens: int = 0

    def summary(self) -> dict[str, float | int | None]:
        """Yield and outcome rates for this device kind."""
        return {
            "owners": self.owners,
            "owner_epochs": self.owner_epochs,
            "observed_channel_fraction": _ratio(self.observed_slots, self.channel_slots),
            "masked_channel_fraction": (
                None
                if self.channel_slots == 0
                else 1.0 - (self.observed_slots / self.channel_slots)
            ),
            "reporting_epoch_fraction": _ratio(self.reporting_epochs, self.owner_epochs),
            "tokens": self.tokens,
            "true_positive_rate": _ratio(self.hazard_tokens, self.hazard_epochs),
            "false_positive_rate": _ratio(self.clean_tokens, self.clean_epochs),
        }


def _ratio(numerator: float, denominator: float) -> float | None:
    """Numerator over denominator, or None when nothing was observed."""
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


@dataclass
class AblationProbe:
    """Drop-one-channel diagnostic over a sample of alarming epochs.

    On a sampled epoch the agent's observed sub-vector is re-scored once per
    observed channel with that channel removed, against the width-corrected cut
    for the reduced vector. ``retained`` counts the alarms that survive the
    removal, so a channel whose retention is near one contributed nothing to
    those alarms and a channel whose retention is near zero was carrying them
    alone.

    Sampling is per epoch, so a channel's counts are of the *alarms it was
    present for*, not of the fleet. Only the instant detector is probed: the
    sequential detector's statistic is path-dependent, and a single-epoch
    re-score would not say what the CUSUM would have done.

    The probe draws from its own generator, so turning a diagnostic on cannot
    move the simulation's random stream and change what it measures.
    """

    channel_set: ChannelSet
    sample_rate: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    alarms_sampled: int = 0
    evaluated: dict[str, int] = field(default_factory=dict)
    retained: dict[str, int] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Whether any epoch will be sampled."""
        return self.sample_rate > 0.0

    def maybe_record(
        self,
        *,
        baseline: BaselineTracker,
        observation: NDArray[np.float64],
        hour: int,
        month: int,
        observed: NDArray[np.bool_] | None,
        reference_threshold: float,
    ) -> None:
        """Re-score one alarming epoch without each observed channel in turn."""
        if not self.enabled or self.rng.random() >= self.sample_rate:
            return
        width = len(self.channel_set)
        mask = np.ones(width, dtype=np.bool_) if observed is None else observed
        columns = np.nonzero(mask)[0]
        if columns.size < 2:
            # Removing the only observed channel leaves nothing to score.
            return
        self.alarms_sampled += 1
        reduced_threshold = threshold_for_dof(reference_threshold, int(columns.size) - 1)
        for column in columns:
            reduced = mask.copy()
            reduced[column] = False
            distance = baseline.mahalanobis_distance(observation, hour, month, reduced)
            name = self.channel_set.names[int(column)]
            self.evaluated[name] = self.evaluated.get(name, 0) + 1
            if distance > reduced_threshold:
                self.retained[name] = self.retained.get(name, 0) + 1

    def summary(self) -> dict[str, object]:
        """Per-channel alarm retention and marginal contribution."""
        channels: dict[str, dict[str, float | int | None]] = {}
        for name in sorted(self.evaluated):
            evaluated = self.evaluated[name]
            retained = self.retained.get(name, 0)
            retention = _ratio(retained, evaluated)
            channels[name] = {
                "alarms_evaluated": evaluated,
                "alarms_retained": retained,
                "alarm_retention": retention,
                "marginal_contribution": None if retention is None else 1.0 - retention,
            }
        return {
            "sample_rate": self.sample_rate,
            "alarms_sampled": self.alarms_sampled,
            "channels": channels,
        }


@dataclass
class DetectionPowerTracker:
    """Accumulates width- and device-stratified detection outcomes.

    Recording is vectorized over the fleet: the simulation fills per-agent
    arrays while it walks its agents anyway, and one call folds a whole step in.
    """

    ablation: AblationProbe | None = None
    alarm_calibration: dict[str, object] | None = None
    zone_threshold_calibration: dict[str, object] | None = None
    _width_cells: dict[str, _WidthCell] = field(
        default_factory=lambda: {label: _WidthCell() for label in WIDTH_BUCKET_LABELS}
    )
    _device_cells: dict[str, _DeviceCell] = field(default_factory=dict)
    _silent_epochs: int = 0
    _hazard_onset_step: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    _latency_recorded: NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )

    def _ensure_capacity(self, size: int) -> None:
        """Grow the per-agent latency state to cover ``size`` agents."""
        if self._hazard_onset_step.size >= size:
            return
        extra = size - self._hazard_onset_step.size
        self._hazard_onset_step = np.concatenate(
            [self._hazard_onset_step, np.full(extra, -1, dtype=np.int64)]
        )
        self._latency_recorded = np.concatenate(
            [self._latency_recorded, np.zeros(extra, dtype=np.bool_)]
        )

    def record_epochs(
        self,
        *,
        step: int,
        widths: NDArray[np.int_],
        emitted: NDArray[np.bool_],
        hazard: NDArray[np.bool_],
    ) -> None:
        """Fold one step of per-agent scoring outcomes into the width buckets.

        ``widths`` is the effective width each agent was scored at, zero for an
        agent that reported nothing. ``hazard`` is the model-side truth that the
        agent was infected or plume-exposed this step, which is an oracle and
        never reaches the protocol.
        """
        if not (widths.shape == emitted.shape == hazard.shape):
            raise ValueError("widths, emitted and hazard must describe the same agents")
        self._ensure_capacity(widths.size)
        reporting = widths > 0
        self._silent_epochs += int(np.count_nonzero(~reporting))
        ids = _bucket_ids(widths[reporting])
        n_buckets = len(WIDTH_BUCKET_LABELS)
        reporting_emitted = emitted[reporting]
        reporting_hazard = hazard[reporting]
        totals = {
            "epochs": np.bincount(ids, minlength=n_buckets),
            "width_sum": np.bincount(ids, weights=widths[reporting], minlength=n_buckets),
            "tokens": np.bincount(ids, weights=reporting_emitted, minlength=n_buckets),
            "hazard_epochs": np.bincount(ids, weights=reporting_hazard, minlength=n_buckets),
            "hazard_tokens": np.bincount(
                ids, weights=reporting_hazard & reporting_emitted, minlength=n_buckets
            ),
            "clean_epochs": np.bincount(ids, weights=~reporting_hazard, minlength=n_buckets),
            "clean_tokens": np.bincount(
                ids, weights=~reporting_hazard & reporting_emitted, minlength=n_buckets
            ),
        }
        for position, label in enumerate(WIDTH_BUCKET_LABELS):
            cell = self._width_cells[label]
            for name, counts in totals.items():
                setattr(cell, name, getattr(cell, name) + int(counts[position]))
        self._record_latencies(step=step, widths=widths, emitted=emitted, hazard=hazard)

    def _record_latencies(
        self,
        *,
        step: int,
        widths: NDArray[np.int_],
        emitted: NDArray[np.bool_],
        hazard: NDArray[np.bool_],
    ) -> None:
        """Time each agent's first token after it became hazard-affected.

        An agent that clears its hazard is re-armed, so a second exposure is
        timed again rather than being silently credited to the first.
        """
        onset = self._hazard_onset_step[: widths.size]
        recorded = self._latency_recorded[: widths.size]
        cleared = ~hazard
        onset[cleared] = -1
        recorded[cleared] = False
        onset[hazard & (onset < 0)] = step
        timed = hazard & emitted & ~recorded & (onset >= 0)
        if not timed.any():
            return
        latencies = step - onset[timed]
        for bucket_id, latency in zip(_bucket_ids(widths[timed]), latencies, strict=True):
            self._width_cells[WIDTH_BUCKET_LABELS[int(bucket_id)]].record_latency(int(latency))
        recorded[timed] = True

    def record_device_epochs(
        self,
        *,
        fleet: DeviceFleet,
        observed: NDArray[np.bool_],
        emitted: NDArray[np.bool_],
        hazard: NDArray[np.bool_],
    ) -> None:
        """Fold one step of per-subsystem reporting yield and outcomes.

        ``observed`` is the *effective* mask actually scored, so a subsystem
        whose battery is flat counts as owned-but-masked rather than absent.
        """
        for position, kind in enumerate(fleet.kinds):
            owned = fleet.ownership[:, position]
            owner_count = int(np.count_nonzero(owned))
            if owner_count == 0:
                continue
            cell = self._device_cells.setdefault(kind.name, _DeviceCell())
            columns = np.asarray(list(fleet.columns_of(kind)), dtype=np.int64)
            owned_observed = observed[np.ix_(owned, columns)]
            owned_emitted = emitted[owned]
            owned_hazard = hazard[owned]
            cell.owners = max(cell.owners, owner_count)
            cell.owner_epochs += owner_count
            cell.channel_slots += owner_count * len(columns)
            cell.observed_slots += int(np.count_nonzero(owned_observed))
            cell.reporting_epochs += int(np.count_nonzero(owned_observed.any(axis=1)))
            cell.tokens += int(np.count_nonzero(owned_emitted))
            cell.hazard_epochs += int(np.count_nonzero(owned_hazard))
            cell.hazard_tokens += int(np.count_nonzero(owned_hazard & owned_emitted))
            cell.clean_epochs += int(np.count_nonzero(~owned_hazard))
            cell.clean_tokens += int(np.count_nonzero(~owned_hazard & owned_emitted))

    @property
    def scored_epochs(self) -> int:
        """Agent-epochs that were scored at one or more channels."""
        return sum(cell.epochs for cell in self._width_cells.values())

    def mean_effective_width(self) -> float | None:
        """Mean number of channels scored per reporting agent-epoch."""
        return _ratio(
            sum(cell.width_sum for cell in self._width_cells.values()),
            self.scored_epochs,
        )

    def summary(self) -> dict[str, object]:
        """Width buckets, device telemetry and any channel ablation."""
        payload: dict[str, object] = {
            "scored_epochs": self.scored_epochs,
            "silent_epochs": self._silent_epochs,
            "mean_effective_width": self.mean_effective_width(),
            "width_buckets": {
                label: self._width_cells[label].summary() for label in WIDTH_BUCKET_LABELS
            },
            "devices": {name: cell.summary() for name, cell in sorted(self._device_cells.items())},
        }
        if self.alarm_calibration is not None:
            payload["alarm_calibration"] = self.alarm_calibration
        if self.zone_threshold_calibration is not None:
            payload["zone_threshold_calibration"] = self.zone_threshold_calibration
        if self.ablation is not None and self.ablation.enabled:
            payload["channel_ablation"] = self.ablation.summary()
        return payload


__all__ = [
    "WIDTH_BUCKET_LABELS",
    "WIDTH_BUCKET_LOWER_BOUNDS",
    "AblationProbe",
    "DetectionPowerConfig",
    "DetectionPowerTracker",
    "width_bucket",
]
