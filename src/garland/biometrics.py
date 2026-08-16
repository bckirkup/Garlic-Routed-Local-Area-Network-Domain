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
    build_profile,
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
from garland.channels import DEFAULT_CHANNEL_SET, ChannelSet

# Residual samples required before the learned covariance replaces the prior.
COVARIANCE_WARMUP_SAMPLES = 5


@dataclass
class BaselineTracker:
    """Exponential time-decay baseline with compressed cyclical deviations.

    Implements B(t) = ∫ X(τ) · e^{-λ(t-τ)} dτ as a running EMA.
    The forgetting rate λ is parameterizable to control privacy:
    higher λ = faster forgetting = more privacy.

    Also maintains long-term cyclical profiles as deviations from the EMA.
    An unlearned bin therefore contributes a zero correction rather than
    pulling the expected baseline toward the origin.

    Parameters
    ----------
    decay_lambda : float
        Exponential decay rate (per 5-min step). Default 0.01 gives an EMA
        half-life of ``ln(2) / 0.01 ≈ 69`` steps, or 5.8 hours.
    seasonal_decay : float
        Decay for cyclical-profile learning. Default 0.001 gives a
        half-life of about 693 updates; elapsed wall-clock time depends on
        how often the relevant hour or month bin is visited.
    channel_set : ChannelSet
        Channel layout of the observations this tracker will receive. All
        state arrays are sized from it, so a wider fleet needs no changes
        here.
    """

    decay_lambda: float = 0.01
    seasonal_decay: float = 0.001
    covariance_prior_strength: float = 1.0
    channel_set: ChannelSet = DEFAULT_CHANNEL_SET
    ema: NDArray[np.float64] = field(init=False)
    circadian_profile: NDArray[np.float64] = field(init=False)
    circadian_counts: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(24, dtype=np.float64)
    )
    monthly_profile: NDArray[np.float64] = field(init=False)
    monthly_counts: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(12, dtype=np.float64)
    )
    covariance_prior: NDArray[np.float64] = field(init=False)
    cov_sum: NDArray[np.float64] = field(init=False)
    n_samples: int = 0

    def __post_init__(self) -> None:
        width = len(self.channel_set)
        self.ema = np.zeros(width, dtype=np.float64)
        self.circadian_profile = np.zeros((24, width), dtype=np.float64)
        self.monthly_profile = np.zeros((12, width), dtype=np.float64)
        self.covariance_prior = np.diag(self.channel_set.prior_variances)
        self.cov_sum = np.zeros((width, width), dtype=np.float64)

    def _profile_is_learned(self, count: float) -> bool:
        """Enable a profile after it has accumulated 5% effective weight."""
        s_alpha = 1.0 - np.exp(-self.seasonal_decay)
        effective_weight = 1.0 - (1.0 - s_alpha) ** max(count - 1.0, 0.0)
        return effective_weight >= 0.05

    def update(self, observation: NDArray[np.float64], hour: int, month: int) -> None:
        """Incorporate a new 5-min observation into the baseline."""
        pre_update_baseline = self.expected_baseline(hour, month)
        residual = observation - pre_update_baseline

        alpha = 1.0 - np.exp(-self.decay_lambda)
        self.ema = (1.0 - alpha) * self.ema + alpha * observation

        s_alpha = 1.0 - np.exp(-self.seasonal_decay)
        h = hour % 24
        self.circadian_profile[h] = (1.0 - s_alpha) * self.circadian_profile[h] + s_alpha * (
            observation - self.ema
        )
        self.circadian_counts[h] += 1

        m = month % 12
        self.monthly_profile[m] = (1.0 - s_alpha) * self.monthly_profile[m] + s_alpha * (
            observation - self.ema
        )
        self.monthly_counts[m] += 1

        self.n_samples += 1
        self.cov_sum += np.outer(residual, residual)

    def expected_baseline(self, hour: int, month: int) -> NDArray[np.float64]:
        """Return expected baseline incorporating circadian + seasonal patterns."""
        h = hour % 24
        m = month % 12
        base = self.ema.copy()
        if self._profile_is_learned(self.circadian_counts[h]):
            base = base + 0.3 * self.circadian_profile[h]
        if self._profile_is_learned(self.monthly_counts[m]):
            base = base + 0.1 * self.monthly_profile[m]
        return base

    def covariance_matrix(self) -> NDArray[np.float64]:
        """Prior-weighted covariance of the scored residual process."""
        if self.n_samples < COVARIANCE_WARMUP_SAMPLES:
            return self.covariance_prior.copy()
        denominator = self.covariance_prior_strength + self.n_samples
        return (self.covariance_prior_strength * self.covariance_prior + self.cov_sum) / denominator

    def mahalanobis_distance(
        self, observation: NDArray[np.float64], hour: int, month: int
    ) -> float:
        """Compute Mahalanobis distance of observation from expected baseline."""
        baseline = self.expected_baseline(hour, month)
        diff = observation - baseline
        cov = self.covariance_matrix()
        cov_reg = cov + np.eye(len(self.channel_set)) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            return float(np.sqrt(np.sum(diff**2)))
        return float(np.sqrt(diff @ cov_inv @ diff))


__all__ = [
    "COVARIANCE_WARMUP_SAMPLES",
    "BaselineTracker",
    "BiometricProfile",
    "SynthesisBackend",
    "build_profile",
    "circadian_factor",
    "generate_observation",
    "generate_observation_custom",
    "generate_observation_neurokit",
    "generate_profiles",
    "seasonal_factor",
]
