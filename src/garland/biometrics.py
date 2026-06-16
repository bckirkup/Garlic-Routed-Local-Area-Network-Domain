"""Biometric synthesis and baseline tracking for GARLAND agents.

Uses NeuroKit2-inspired statistical principles to generate discrete
5-minute aggregate biometric vectors without continuous waveform storage.
Implements exponential time-decay kernel for adaptive baseline learning/forgetting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class BiometricProfile:
    """Resting biometric profile for a single agent.

    Values drawn from physiologically plausible distributions matching
    OpenWearables schema conventions.
    """

    resting_hr: float  # beats/min
    resting_hrv: float  # RMSSD ms
    resting_rr: float  # breaths/min
    resting_temp: float  # °C core body temp
    # Circadian amplitude factors
    hr_circadian_amp: float = 0.0
    rr_circadian_amp: float = 0.0
    temp_circadian_amp: float = 0.0


def generate_profiles(n: int, rng: np.random.Generator) -> list[BiometricProfile]:
    """Generate physiologically plausible biometric profiles for n agents.

    Distributions are based on published population norms:
    - HR: mean 72, std 8 bpm
    - HRV (RMSSD): mean 42, std 15 ms
    - RR: mean 15, std 2 breaths/min
    - Temp: mean 36.8, std 0.3 °C
    """
    hrs = rng.normal(72.0, 8.0, n).clip(50, 110)
    hrvs = rng.normal(42.0, 15.0, n).clip(10, 120)
    rrs = rng.normal(15.0, 2.0, n).clip(8, 25)
    temps = rng.normal(36.8, 0.3, n).clip(35.5, 38.0)
    hr_ca = rng.uniform(3.0, 8.0, n)
    rr_ca = rng.uniform(0.5, 2.0, n)
    temp_ca = rng.uniform(0.2, 0.5, n)

    return [
        BiometricProfile(
            resting_hr=float(hrs[i]),
            resting_hrv=float(hrvs[i]),
            resting_rr=float(rrs[i]),
            resting_temp=float(temps[i]),
            hr_circadian_amp=float(hr_ca[i]),
            rr_circadian_amp=float(rr_ca[i]),
            temp_circadian_amp=float(temp_ca[i]),
        )
        for i in range(n)
    ]


def circadian_factor(hour_of_day: float) -> float:
    """Circadian modulation factor [-1, 1] peaking around 14:00 (2 PM)."""
    return float(np.sin(2 * np.pi * (hour_of_day - 2.0) / 24.0))


def seasonal_factor(day_of_year: int) -> float:
    """Seasonal baseline shift factor [-1, 1], peak in summer (day 172)."""
    return float(np.sin(2 * np.pi * (day_of_year - 80) / 365.0))


@dataclass
class BaselineTracker:
    """Exponential time-decay baseline with compressed 24h profile.

    Implements B(t) = ∫ X(τ) · e^{-λ(t-τ)} dτ as a running EMA.
    The forgetting rate λ is parameterizable to control privacy:
    higher λ = faster forgetting = more privacy.

    Also maintains a long-term cyclical profile (24 bins for circadian,
    12 bins for seasonal/monthly patterns).

    Parameters
    ----------
    decay_lambda : float
        Exponential decay rate (per 5-min step). Default 0.01 → ~6.9h half-life.
    seasonal_decay : float
        Decay for seasonal learning. Default 0.001 → ~57 day half-life.
    """

    decay_lambda: float = 0.01
    seasonal_decay: float = 0.001
    # Running EMA state (4-dim: HR, HRV, RR, Temp)
    ema: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    # 24-hour circadian profile (24 hourly bins × 4 params)
    circadian_profile: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((24, 4), dtype=np.float64)
    )
    circadian_counts: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(24, dtype=np.float64)
    )
    # 12-month seasonal profile
    monthly_profile: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((12, 4), dtype=np.float64)
    )
    monthly_counts: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(12, dtype=np.float64)
    )
    # Running covariance estimate for Mahalanobis distance
    cov_sum: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4, dtype=np.float64) * 10.0
    )
    n_samples: int = 1

    def update(self, observation: NDArray[np.float64], hour: int, month: int) -> None:
        """Incorporate a new 5-min observation into the baseline."""
        alpha = 1.0 - np.exp(-self.decay_lambda)
        self.ema = (1.0 - alpha) * self.ema + alpha * observation

        # Circadian bin update with exponential weighting
        s_alpha = 1.0 - np.exp(-self.seasonal_decay)
        h = hour % 24
        self.circadian_profile[h] = (
            (1.0 - s_alpha) * self.circadian_profile[h] + s_alpha * observation
        )
        self.circadian_counts[h] += 1

        # Monthly profile
        m = month % 12
        self.monthly_profile[m] = (
            (1.0 - s_alpha) * self.monthly_profile[m] + s_alpha * observation
        )
        self.monthly_counts[m] += 1

        # Incremental covariance estimate
        self.n_samples += 1
        diff = observation - self.ema
        self.cov_sum += np.outer(diff, diff)

    def expected_baseline(self, hour: int, month: int) -> NDArray[np.float64]:
        """Return expected baseline incorporating circadian + seasonal patterns."""
        h = hour % 24
        m = month % 12
        base = self.ema.copy()
        if self.circadian_counts[h] > 5:
            base = 0.7 * base + 0.3 * self.circadian_profile[h]
        if self.monthly_counts[m] > 10:
            base = 0.9 * base + 0.1 * self.monthly_profile[m]
        return base

    def covariance_matrix(self) -> NDArray[np.float64]:
        """Estimated covariance of deviations from baseline."""
        if self.n_samples < 5:
            return np.eye(4, dtype=np.float64) * 10.0
        return self.cov_sum / self.n_samples

    def mahalanobis_distance(
        self, observation: NDArray[np.float64], hour: int, month: int
    ) -> float:
        """Compute Mahalanobis distance of observation from expected baseline."""
        baseline = self.expected_baseline(hour, month)
        diff = observation - baseline
        cov = self.covariance_matrix()
        # Regularize to avoid singularity
        cov_reg = cov + np.eye(4) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            return float(np.sqrt(np.sum(diff**2)))
        return float(np.sqrt(diff @ cov_inv @ diff))


def generate_observation(
    profile: BiometricProfile,
    hour_of_day: float,
    day_of_year: int,
    rng: np.random.Generator,
    activity_level: float = 0.0,
) -> NDArray[np.float64]:
    """Generate a single 5-minute aggregate biometric vector.

    Parameters
    ----------
    profile : BiometricProfile
        Agent's resting biometric parameters.
    hour_of_day : float
        Current hour [0, 24).
    day_of_year : int
        Day of year [1, 365].
    rng : np.random.Generator
        Random number generator.
    activity_level : float
        Physical activity factor [0, 1]. 0 = resting, 1 = peak exercise.
    """
    circ = circadian_factor(hour_of_day)
    seas = seasonal_factor(day_of_year)

    hr = (
        profile.resting_hr
        + profile.hr_circadian_amp * circ
        + activity_level * 40.0
        + rng.normal(0, 2.0)
    )
    hrv = (
        profile.resting_hrv
        - activity_level * 20.0
        + rng.normal(0, 5.0)
    )
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
