"""Sensor channel registry for GARLAND observation vectors.

An observation is a vector of *derived per-epoch features*, one entry per
:class:`Channel`, ordered by the agent's :class:`ChannelSet`. Devices never
export waveforms; a channel is whatever a device can compute on-body and
report for a five-minute epoch.

Every consumer of an observation (synthesis, baseline tracking, sequential
detection, anomaly classification, hazard and confounder deltas, Open
Wearables export) addresses entries by channel *name* through this registry
rather than by position, so a wider channel set requires no changes to those
consumers.

``CORE_VITALS`` is the historical four-channel set (HR, HRV RMSSD, RR, core
temperature) and remains the default. Its parameters reproduce the previous
hard-coded synthesis exactly, including RNG draw order: channel noise is drawn
in channel order, and a channel with a degenerate circadian-amplitude range
draws nothing at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

# Number of exceeded channels at or above which an excursion is reported as
# multi-system rather than attributed to one physiological system.
MULTI_SYSTEM_MIN_CHANNELS = 3


class ChannelSystem(str, Enum):
    """Physiological system a channel measures.

    Classification rules are written against systems, not channel names, so
    adding a second cardiac or respiratory channel does not require new rules.
    """

    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    THERMAL = "thermal"


@dataclass(frozen=True)
class Channel:
    """One named feature channel and the parameters its consumers need.

    Parameters
    ----------
    name, unit, system
        Identity of the channel. ``name`` is the key used by every
        name-addressed API in this package.
    resting_mean, resting_sd, resting_min, resting_max
        Population distribution of the per-person resting value.
    circadian_amp_min, circadian_amp_max
        Per-person circadian amplitude range. A degenerate range (min equal to
        max) means the channel has no circadian term and draws no random
        amplitude.
    circadian_scale, seasonal_coefficient, activity_coefficient
        Multipliers on the shared circadian factor, seasonal factor, and the
        agent's activity level.
    noise_sd
        Standard deviation of the per-epoch measurement and physiological
        noise.
    floor
        Optional lower clamp applied after synthesis (RMSSD cannot be zero).
    deviation_threshold
        Absolute deviation from baseline at which this channel counts as
        excursion-positive during classification.
    quiet_fraction
        Fraction of ``deviation_threshold`` below which the channel counts as
        quiet. Used by rules that require a channel to be *un*-perturbed, such
        as respiratory distress without fever.
    prior_variance
        Diagonal entry of the cold-start covariance prior.
    openwearables_type
        Open Wearables timeseries type, or ``None`` for channels with no
        equivalent in that schema (those are omitted from exports).
    """

    name: str
    unit: str
    system: ChannelSystem
    resting_mean: float
    resting_sd: float
    resting_min: float
    resting_max: float
    noise_sd: float
    deviation_threshold: float
    circadian_amp_min: float = 0.0
    circadian_amp_max: float = 0.0
    circadian_scale: float = 1.0
    seasonal_coefficient: float = 0.0
    activity_coefficient: float = 0.0
    floor: float | None = None
    quiet_fraction: float = 1.0
    prior_variance: float = 10.0
    openwearables_type: str | None = None

    @property
    def has_circadian_amplitude(self) -> bool:
        """Whether this channel draws a per-person circadian amplitude."""
        return self.circadian_amp_min != self.circadian_amp_max


HEART_RATE = Channel(
    name="heart_rate",
    unit="bpm",
    system=ChannelSystem.CARDIAC,
    resting_mean=72.0,
    resting_sd=8.0,
    resting_min=50.0,
    resting_max=110.0,
    noise_sd=2.0,
    deviation_threshold=10.0,
    circadian_amp_min=3.0,
    circadian_amp_max=8.0,
    activity_coefficient=40.0,
    openwearables_type="heart_rate",
)

HRV_RMSSD = Channel(
    name="hrv_rmssd",
    unit="ms",
    system=ChannelSystem.CARDIAC,
    resting_mean=42.0,
    resting_sd=15.0,
    resting_min=10.0,
    resting_max=120.0,
    noise_sd=5.0,
    deviation_threshold=10.0,
    activity_coefficient=-20.0,
    floor=5.0,
    openwearables_type="heart_rate_variability_rmssd",
)

RESPIRATORY_RATE = Channel(
    name="respiratory_rate",
    unit="brpm",
    system=ChannelSystem.RESPIRATORY,
    resting_mean=15.0,
    resting_sd=2.0,
    resting_min=8.0,
    resting_max=25.0,
    noise_sd=1.0,
    deviation_threshold=4.0,
    circadian_amp_min=0.5,
    circadian_amp_max=2.0,
    activity_coefficient=8.0,
    openwearables_type="respiratory_rate",
)

BODY_TEMPERATURE = Channel(
    name="body_temperature",
    unit="°C",
    system=ChannelSystem.THERMAL,
    resting_mean=36.8,
    resting_sd=0.3,
    resting_min=35.5,
    resting_max=38.0,
    noise_sd=0.1,
    deviation_threshold=0.8,
    circadian_amp_min=0.2,
    circadian_amp_max=0.5,
    circadian_scale=0.3,
    seasonal_coefficient=0.1,
    activity_coefficient=0.5,
    # 0.5 °C over a 0.8 °C threshold: the "no fever" arm of respiratory rules.
    quiet_fraction=0.625,
    openwearables_type="body_temperature",
)


@dataclass(frozen=True)
class ChannelSet:
    """An ordered set of channels defining one observation vector layout."""

    channels: tuple[Channel, ...]
    _index: dict[str, int] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("a channel set must contain at least one channel")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate channel names in channel set: {names}")
        self._index.update({name: position for position, name in enumerate(names)})

    def __len__(self) -> int:
        return len(self.channels)

    @property
    def names(self) -> tuple[str, ...]:
        """Channel names in vector order."""
        return tuple(channel.name for channel in self.channels)

    def index(self, name: str) -> int:
        """Vector position of ``name``."""
        try:
            return self._index[name]
        except KeyError as exc:
            raise KeyError(
                f"channel {name!r} is not in this channel set; have {self.names}"
            ) from exc

    def has(self, name: str) -> bool:
        """Whether ``name`` is present in this channel set."""
        return name in self._index

    def get(self, name: str) -> Channel:
        """Return the channel named ``name``."""
        return self.channels[self.index(name)]

    def zeros(self) -> NDArray[np.float64]:
        """An all-zero vector with this set's width."""
        return np.zeros(len(self.channels), dtype=np.float64)

    def delta(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        """Build an additive perturbation vector from named channel deltas.

        Channels absent from ``values`` are zero. Unknown names raise, so a
        signature written for a channel the fleet does not carry is a
        configuration error rather than a silently dropped effect.
        """
        vector = self.zeros()
        for name, value in values.items():
            vector[self.index(name)] = value
        return vector

    def system_indices(self, system: ChannelSystem) -> tuple[int, ...]:
        """Vector positions of every channel measuring ``system``."""
        return tuple(
            position for position, channel in enumerate(self.channels) if channel.system == system
        )

    @property
    def deviation_thresholds(self) -> NDArray[np.float64]:
        """Per-channel excursion thresholds in vector order."""
        return np.array(
            [channel.deviation_threshold for channel in self.channels], dtype=np.float64
        )

    @property
    def prior_variances(self) -> NDArray[np.float64]:
        """Per-channel cold-start covariance prior diagonal, in vector order."""
        return np.array([channel.prior_variance for channel in self.channels], dtype=np.float64)

    def with_channels(self, *channels: Channel) -> ChannelSet:
        """Return a wider set with ``channels`` appended."""
        return ChannelSet(self.channels + channels)


CORE_VITALS = ChannelSet((HEART_RATE, HRV_RMSSD, RESPIRATORY_RATE, BODY_TEMPERATURE))

DEFAULT_CHANNEL_SET = CORE_VITALS


__all__ = [
    "BODY_TEMPERATURE",
    "CORE_VITALS",
    "DEFAULT_CHANNEL_SET",
    "HEART_RATE",
    "HRV_RMSSD",
    "MULTI_SYSTEM_MIN_CHANNELS",
    "RESPIRATORY_RATE",
    "Channel",
    "ChannelSet",
    "ChannelSystem",
]
