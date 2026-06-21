"""Biometric baseline tracking for GARLAND agents.

Observation synthesis backends live in ``garland.biometric_synthesis``.
Profiles and circadian helpers live in ``garland.biometric_profiles``.
See ``docs/BIOMETRICS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.biometric_profiles import (
    BiometricProfile,
    circadian_factor,
    generate_profiles,
    seasonal_factor,
)
from garland.biometric_synthesis import (
    SynthesisBackend,
    generate_observation,
    generate_observation_custom,
    generate_observation_neurokit,
)


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
    ema: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    circadian_profile: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((24, 4), dtype=np.float64)
    )
    circadian_counts: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(24, dtype=np.float64)
    )
    monthly_profile: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((12, 4), dtype=np.float64)
    )
    monthly_counts: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(12, dtype=np.float64)
    )
    cov_sum: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4, dtype=np.float64) * 10.0
    )
    n_samples: int = 1

    def update(self, observation: NDArray[np.float64], hour: int, month: int) -> None:
        """Incorporate a new 5-min observation into the baseline."""
        alpha = 1.0 - np.exp(-self.decay_lambda)
        self.ema = (1.0 - alpha) * self.ema + alpha * observation

        s_alpha = 1.0 - np.exp(-self.seasonal_decay)
        h = hour % 24
        self.circadian_profile[h] = (
            (1.0 - s_alpha) * self.circadian_profile[h] + s_alpha * observation
        )
        self.circadian_counts[h] += 1

        m = month % 12
        self.monthly_profile[m] = (
            (1.0 - s_alpha) * self.monthly_profile[m] + s_alpha * observation
        )
        self.monthly_counts[m] += 1

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
        cov_reg = cov + np.eye(4) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            return float(np.sqrt(np.sum(diff**2)))
        return float(np.sqrt(diff @ cov_inv @ diff))


__all__ = [
    "BaselineTracker",
    "BiometricProfile",
    "SynthesisBackend",
    "circadian_factor",
    "generate_observation",
    "generate_observation_custom",
    "generate_observation_neurokit",
    "generate_profiles",
    "seasonal_factor",
]
