"""Biometric observation synthesis backends for GARLAND.

The default ``custom`` backend uses fast NumPy synthesis suitable for
city-scale runs. The optional ``neurokit`` backend generates ECG/RSP
signals via NeuroKit2 and extracts aggregate vitals for validation
and research subsets (much slower — not for 250K-agent production runs).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from garland.biometric_profiles import (
    BiometricProfile,
    circadian_factor,
    seasonal_factor,
)

SynthesisBackend = Literal["custom", "neurokit"]

# Shorter window keeps NeuroKit2 validation runs practical in tests.
NEUROKIT_DEFAULT_WINDOW_SECONDS = 60.0
NEUROKIT_SAMPLING_RATE = 50


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

    hr = (
        profile.resting_hr
        + profile.hr_circadian_amp * circ
        + activity_level * 40.0
        + rng.normal(0, 2.0)
    )
    hrv = profile.resting_hrv - activity_level * 20.0 + rng.normal(0, 5.0)
    rr = (
        profile.resting_rr
        + profile.rr_circadian_amp * circ
        + activity_level * 8.0
        + rng.normal(0, 1.0)
    )
    temp = (
        profile.resting_temp
        + profile.temp_circadian_amp * circ * 0.3
        + seas * 0.1
        + activity_level * 0.5
        + rng.normal(0, 0.1)
    )

    return np.array([hr, max(5.0, hrv), rr, temp], dtype=np.float64)


def _require_neurokit2():
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise ImportError(
            "NeuroKit2 synthesis requires optional dependencies. "
            "Install with: pip install -e \".[biosignals]\""
        ) from exc
    return nk


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
    Respiratory rate comes from ``rsp_simulate`` + ``rsp_process``.
    Core temperature uses the custom circadian model (NeuroKit2 has no
    body-temperature simulator in this path).
    """
    nk = _require_neurokit2()

    circ = circadian_factor(hour_of_day)
    seas = seasonal_factor(day_of_year)
    target_hr = (
        profile.resting_hr
        + profile.hr_circadian_amp * circ
        + activity_level * 40.0
    )
    target_rr = (
        profile.resting_rr
        + profile.rr_circadian_amp * circ
        + activity_level * 8.0
    )
    temp = (
        profile.resting_temp
        + profile.temp_circadian_amp * circ * 0.3
        + seas * 0.1
        + activity_level * 0.5
        + rng.normal(0, 0.1)
    )

    seed = int(rng.integers(0, 2**31))
    duration = max(int(window_seconds), 10)
    ecg = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=float(np.clip(target_hr, 50, 110)),
        random_state=seed,
    )
    ecg_signals, _ = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    hr = float(ecg_signals["ECG_Rate"].mean())

    hrv = profile.resting_hrv
    if len(ecg_signals) > sampling_rate * 5:
        hrv_stats = nk.hrv_time(ecg_signals, sampling_rate=sampling_rate)
        hrv = float(hrv_stats["HRV_RMSSD"].iloc[0])

    rsp = nk.rsp_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        respiratory_rate=float(np.clip(target_rr, 8, 25)),
        random_state=seed + 1,
    )
    rsp_signals, _ = nk.rsp_process(rsp, sampling_rate=sampling_rate)
    rr = float(rsp_signals["RSP_Rate"].mean())

    return np.array([hr, max(5.0, hrv), rr, temp], dtype=np.float64)


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
        return generate_observation_custom(
            profile, hour_of_day, day_of_year, rng, activity_level
        )
    raise ValueError(f"Unknown biometric synthesis backend {backend!r}")
