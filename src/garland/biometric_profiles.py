"""Biometric profile definitions and population generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from garland.channels import DEFAULT_CHANNEL_SET, ChannelSet


@dataclass
class BiometricProfile:
    """Resting biometric profile for a single agent.

    ``resting`` and ``circadian_amp`` are vectors indexed by ``channel_set``,
    drawn from the population distributions declared on each
    :class:`~garland.channels.Channel`.
    """

    resting: NDArray[np.float64]
    circadian_amp: NDArray[np.float64]
    channel_set: ChannelSet = DEFAULT_CHANNEL_SET

    def __post_init__(self) -> None:
        width = len(self.channel_set)
        if len(self.resting) != width or len(self.circadian_amp) != width:
            raise ValueError(
                f"profile vectors must have {width} entries for channel set "
                f"{self.channel_set.names}, got {len(self.resting)} resting and "
                f"{len(self.circadian_amp)} circadian"
            )

    def resting_value(self, channel: str) -> float:
        """Resting value of one named channel."""
        return float(self.resting[self.channel_set.index(channel)])

    def circadian_amplitude(self, channel: str) -> float:
        """Circadian amplitude of one named channel."""
        return float(self.circadian_amp[self.channel_set.index(channel)])


def build_profile(
    resting: Mapping[str, float] | None = None,
    circadian_amp: Mapping[str, float] | None = None,
    channel_set: ChannelSet = DEFAULT_CHANNEL_SET,
) -> BiometricProfile:
    """Build one profile from named values, defaulting to channel resting means.

    Convenience constructor for tests, examples, and fixed scenarios: unnamed
    channels take their population mean resting value and no circadian
    amplitude. Unknown channel names raise.
    """
    resting_vector = np.array(
        [channel.resting_mean for channel in channel_set.channels], dtype=np.float64
    )
    amp_vector = channel_set.zeros()
    for name, value in (resting or {}).items():
        resting_vector[channel_set.index(name)] = value
    for name, value in (circadian_amp or {}).items():
        amp_vector[channel_set.index(name)] = value
    return BiometricProfile(
        resting=resting_vector,
        circadian_amp=amp_vector,
        channel_set=channel_set,
    )


def generate_profiles(
    n: int,
    rng: np.random.Generator,
    channel_set: ChannelSet = DEFAULT_CHANNEL_SET,
) -> list[BiometricProfile]:
    """Generate physiologically plausible biometric profiles for n agents.

    Resting values are drawn for every channel in vector order, then circadian
    amplitudes for the channels that declare a non-degenerate amplitude range.
    """
    width = len(channel_set)
    resting = np.empty((n, width), dtype=np.float64)
    amplitudes = np.zeros((n, width), dtype=np.float64)

    for position, channel in enumerate(channel_set.channels):
        resting[:, position] = rng.normal(channel.resting_mean, channel.resting_sd, n).clip(
            channel.resting_min, channel.resting_max
        )
    for position, channel in enumerate(channel_set.channels):
        if channel.has_circadian_amplitude:
            amplitudes[:, position] = rng.uniform(
                channel.circadian_amp_min, channel.circadian_amp_max, n
            )
        else:
            amplitudes[:, position] = channel.circadian_amp_min

    return [
        BiometricProfile(
            resting=resting[i].copy(),
            circadian_amp=amplitudes[i].copy(),
            channel_set=channel_set,
        )
        for i in range(n)
    ]


def circadian_factor(hour_of_day: float) -> float:
    """Circadian modulation factor [-1, 1] peaking around 14:00 (2 PM)."""
    return float(np.sin(2 * np.pi * (hour_of_day - 2.0) / 24.0))


def seasonal_factor(day_of_year: int) -> float:
    """Seasonal baseline shift factor [-1, 1], peak in summer (day 172)."""
    return float(np.sin(2 * np.pi * (day_of_year - 80) / 365.0))
