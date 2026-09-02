"""Biometric observation synthesis backends for GARLAND.

The default ``custom`` backend uses fast NumPy synthesis suitable for
city-scale runs. The optional ``neurokit`` backend generates ECG/RSP
signals via NeuroKit2 and extracts aggregate vitals for validation
and research subsets (much slower — not for 250K-agent production runs).

Both backends synthesize whatever channel set the profile carries. The
``neurokit`` backend derives the cardiac and respiratory channels it knows how
to simulate and falls back to the custom formula for every other channel.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from garland.biometric_profiles import (
    BiometricProfile,
    circadian_factor,
    seasonal_factor,
)
from garland.channels import Channel, ChannelSet

SynthesisBackend = Literal["custom", "neurokit"]

# Shorter window keeps NeuroKit2 validation runs practical in tests.
NEUROKIT_DEFAULT_WINDOW_SECONDS = 60.0
NEUROKIT_SAMPLING_RATE = 50

# Channels NeuroKit2 can derive from simulated ECG/RSP waveforms.
NEUROKIT_HR_CHANNEL = "heart_rate"
NEUROKIT_HRV_CHANNEL = "hrv_rmssd"
NEUROKIT_RR_CHANNEL = "respiratory_rate"
NEUROKIT_DERIVED_CHANNELS = frozenset(
    {NEUROKIT_HR_CHANNEL, NEUROKIT_HRV_CHANNEL, NEUROKIT_RR_CHANNEL}
)

# Heart rate used to drive ECG simulation when a channel set carries HRV but no
# heart-rate channel.
HR_SIMULATION_FALLBACK = 72.0


@lru_cache(maxsize=None)
def _channel_vectors(
    channels: tuple[Channel, ...],
) -> tuple[NDArray[np.float64], ...]:
    """Return channel coefficients reused by fleet-level custom synthesis."""
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            [channel.circadian_scale for channel in channels],
            [channel.seasonal_coefficient for channel in channels],
            [channel.activity_coefficient for channel in channels],
            [channel.noise_sd for channel in channels],
            [channel.floor if channel.floor is not None else -np.inf for channel in channels],
        )
    )


def _deterministic_level(
    profile: BiometricProfile,
    position: int,
    channel: Channel,
    circ: float,
    seas: float,
    activity_level: float,
) -> float:
    """Noise-free synthesis level for one channel."""
    return (
        profile.resting[position]
        + (profile.circadian_amp[position] * circ) * channel.circadian_scale
        + seas * channel.seasonal_coefficient
        + activity_level * channel.activity_coefficient
    )


def _synthesize_channel(
    profile: BiometricProfile,
    position: int,
    channel: Channel,
    circ: float,
    seas: float,
    activity_level: float,
    rng: np.random.Generator,
) -> float:
    """Draw one channel value, including its noise term and floor."""
    value = _deterministic_level(
        profile, position, channel, circ, seas, activity_level
    ) + rng.normal(0, channel.noise_sd)
    if channel.floor is not None:
        return max(channel.floor, value)
    return value


def generate_observation_custom(
    profile: BiometricProfile,
    hour_of_day: float,
    day_of_year: int,
    rng: np.random.Generator,
    activity_level: float = 0.0,
) -> NDArray[np.float64]:
    """Fast NumPy synthesis of a 5-minute aggregate biometric vector."""
    circ = circadian_factor(hour_of_day)
    seas = seasonal_factor(day_of_year)
    channel_set = profile.channel_set

    observation = channel_set.zeros()
    for position, channel in enumerate(channel_set.channels):
        observation[position] = _synthesize_channel(
            profile, position, channel, circ, seas, activity_level, rng
        )
    return observation


def generate_observations_batch(
    resting: NDArray[np.float64],
    circadian_amp: NDArray[np.float64],
    channel_set: ChannelSet,
    hour_of_day: float,
    day_of_year: int,
    rng: np.random.Generator,
    activity_levels: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Generate custom observations for a fleet in one vectorized draw."""
    if resting.ndim != 2 or circadian_amp.shape != resting.shape:
        raise ValueError("resting and circadian_amp must have matching 2-D shapes")
    if activity_levels.shape != (resting.shape[0],):
        raise ValueError("activity_levels must have one value per profile")
    if resting.shape[1] != len(channel_set):
        raise ValueError("profile width must match the channel set")

    (
        circadian_scale,
        seasonal_coefficient,
        activity_coefficient,
        noise_sd,
        floors,
    ) = _channel_vectors(channel_set.channels)
    deterministic_level = (
        resting
        + circadian_amp * circadian_factor(hour_of_day) * circadian_scale
        + seasonal_factor(day_of_year) * seasonal_coefficient
        + activity_levels[:, None] * activity_coefficient
    )
    observation = deterministic_level + rng.normal(
        0.0,
        noise_sd,
        size=resting.shape,
    )
    return np.maximum(observation, floors).astype(np.float64, copy=False)


def _require_neurokit2():
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise ImportError(
            "NeuroKit2 synthesis requires optional dependencies. "
            'Install with: pip install -e ".[biosignals]"'
        ) from exc
    return nk


def _fill_cardiac_channels(
    nk,
    observation: NDArray[np.float64],
    profile: BiometricProfile,
    *,
    target_hr: float,
    duration: int,
    sampling_rate: int,
    seed: int,
) -> None:
    """Fill the heart-rate and HRV channels from one simulated ECG window."""
    channel_set = profile.channel_set
    ecg = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=float(np.clip(target_hr, 50, 110)),
        random_state=seed,
    )
    ecg_signals, _ = nk.ecg_process(ecg, sampling_rate=sampling_rate)

    if channel_set.has(NEUROKIT_HR_CHANNEL):
        observation[channel_set.index(NEUROKIT_HR_CHANNEL)] = float(ecg_signals["ECG_Rate"].mean())

    if not channel_set.has(NEUROKIT_HRV_CHANNEL):
        return
    hrv_position = channel_set.index(NEUROKIT_HRV_CHANNEL)
    hrv = float(profile.resting[hrv_position])
    if len(ecg_signals) > sampling_rate * 5:
        hrv_stats = nk.hrv_time(ecg_signals, sampling_rate=sampling_rate)
        hrv = float(hrv_stats["HRV_RMSSD"].iloc[0])
    floor = channel_set.get(NEUROKIT_HRV_CHANNEL).floor
    observation[hrv_position] = hrv if floor is None else max(floor, hrv)


def generate_observation_neurokit(
    profile: BiometricProfile,
    hour_of_day: float,
    day_of_year: int,
    rng: np.random.Generator,
    activity_level: float = 0.0,
    window_seconds: float = NEUROKIT_DEFAULT_WINDOW_SECONDS,
    sampling_rate: int = NEUROKIT_SAMPLING_RATE,
) -> NDArray[np.float64]:
    """NeuroKit2-backed synthesis: simulate ECG/RSP and extract aggregates.

    Heart rate and HRV (RMSSD) come from ``ecg_simulate`` + ``ecg_process``.
    Respiratory rate comes from ``rsp_simulate`` + ``rsp_process``. Every other
    channel — core temperature in the default set — uses the custom formula,
    since NeuroKit2 has no simulator for it in this path.
    """
    nk = _require_neurokit2()

    circ = circadian_factor(hour_of_day)
    seas = seasonal_factor(day_of_year)
    channel_set = profile.channel_set

    observation = channel_set.zeros()
    for position, channel in enumerate(channel_set.channels):
        if channel.name in NEUROKIT_DERIVED_CHANNELS:
            continue
        observation[position] = _synthesize_channel(
            profile, position, channel, circ, seas, activity_level, rng
        )

    targets = {
        name: _deterministic_level(
            profile, channel_set.index(name), channel_set.get(name), circ, seas, activity_level
        )
        for name in (NEUROKIT_HR_CHANNEL, NEUROKIT_RR_CHANNEL)
        if channel_set.has(name)
    }

    seed = int(rng.integers(0, 2**31))
    duration = max(int(window_seconds), 10)

    if channel_set.has(NEUROKIT_HR_CHANNEL) or channel_set.has(NEUROKIT_HRV_CHANNEL):
        _fill_cardiac_channels(
            nk,
            observation,
            profile,
            target_hr=targets.get(NEUROKIT_HR_CHANNEL, HR_SIMULATION_FALLBACK),
            duration=duration,
            sampling_rate=sampling_rate,
            seed=seed,
        )

    if channel_set.has(NEUROKIT_RR_CHANNEL):
        rsp = nk.rsp_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            respiratory_rate=float(np.clip(targets[NEUROKIT_RR_CHANNEL], 8, 25)),
            random_state=seed + 1,
        )
        rsp_signals, _ = nk.rsp_process(rsp, sampling_rate=sampling_rate)
        observation[channel_set.index(NEUROKIT_RR_CHANNEL)] = float(rsp_signals["RSP_Rate"].mean())

    return observation


def generate_observation(
    profile: BiometricProfile,
    hour_of_day: float,
    day_of_year: int,
    rng: np.random.Generator,
    activity_level: float = 0.0,
    backend: SynthesisBackend = "custom",
    neurokit_window_seconds: float = NEUROKIT_DEFAULT_WINDOW_SECONDS,
) -> NDArray[np.float64]:
    """Generate a biometric observation using the requested synthesis backend."""
    if backend == "neurokit":
        return generate_observation_neurokit(
            profile,
            hour_of_day,
            day_of_year,
            rng,
            activity_level=activity_level,
            window_seconds=neurokit_window_seconds,
        )
    if backend == "custom":
        return generate_observation_custom(profile, hour_of_day, day_of_year, rng, activity_level)
    raise ValueError(f"Unknown biometric synthesis backend {backend!r}")


__all__ = [
    "generate_observation",
    "generate_observation_custom",
    "generate_observation_neurokit",
    "generate_observations_batch",
]
