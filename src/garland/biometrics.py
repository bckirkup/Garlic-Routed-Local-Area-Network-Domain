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

    Channels a device did not report this epoch are addressed with an
    ``observed`` mask: masked entries leave their baseline, cyclical profiles and
    covariance untouched instead of being learned as zeros, and scoring uses the
    observed sub-vector. Learning counts are therefore per channel as well as
    per bin, so a duty-cycled channel enables its own cyclical correction only
    once *it* has been seen enough.
    """

    decay_lambda: float = 0.01
    seasonal_decay: float = 0.001
    covariance_prior_strength: float = 1.0
    channel_set: ChannelSet = DEFAULT_CHANNEL_SET
    ema: NDArray[np.float64] = field(init=False)
    circadian_profile: NDArray[np.float64] = field(init=False)
    circadian_counts: NDArray[np.float64] = field(init=False)
    monthly_profile: NDArray[np.float64] = field(init=False)
    monthly_counts: NDArray[np.float64] = field(init=False)
    covariance_prior: NDArray[np.float64] = field(init=False)
    cov_sum: NDArray[np.float64] = field(init=False)
    cov_counts: NDArray[np.float64] = field(init=False)
    n_samples: int = 0

    def __post_init__(self) -> None:
        width = len(self.channel_set)
        self.ema = np.zeros(width, dtype=np.float64)
        self.circadian_profile = np.zeros((24, width), dtype=np.float64)
        self.circadian_counts = np.ones((24, width), dtype=np.float64)
        self.monthly_profile = np.zeros((12, width), dtype=np.float64)
        self.monthly_counts = np.ones((12, width), dtype=np.float64)
        self.covariance_prior = np.diag(self.channel_set.prior_variances)
        self.cov_sum = np.zeros((width, width), dtype=np.float64)
        self.cov_counts = np.zeros((width, width), dtype=np.float64)

    def _observed_mask(self, observed: NDArray[np.bool_] | None) -> NDArray[np.bool_]:
        width = len(self.channel_set)
        if observed is None:
            return np.ones(width, dtype=np.bool_)
        if observed.shape != (width,):
            raise ValueError(
                f"observed mask has {observed.shape} entries but the channel set has {width}"
            )
        return observed

    def _profile_is_learned(self, counts: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Enable a profile after it has accumulated 5% effective weight."""
        s_alpha = 1.0 - np.exp(-self.seasonal_decay)
        effective_weight = 1.0 - (1.0 - s_alpha) ** np.maximum(counts - 1.0, 0.0)
        return effective_weight >= 0.05

    def update(
        self,
        observation: NDArray[np.float64],
        hour: int,
        month: int,
        observed: NDArray[np.bool_] | None = None,
    ) -> None:
        """Incorporate a new 5-min observation into the baseline."""
        mask = self._observed_mask(observed)
        if not mask.any():
            return
        pre_update_baseline = self.expected_baseline(hour, month)
        residual = np.where(mask, observation - pre_update_baseline, 0.0)

        alpha = 1.0 - np.exp(-self.decay_lambda)
        self.ema = np.where(mask, (1.0 - alpha) * self.ema + alpha * observation, self.ema)

        s_alpha = 1.0 - np.exp(-self.seasonal_decay)
        deviation = observation - self.ema
        h = hour % 24
        self.circadian_profile[h] = np.where(
            mask,
            (1.0 - s_alpha) * self.circadian_profile[h] + s_alpha * deviation,
            self.circadian_profile[h],
        )
        self.circadian_counts[h] += mask

        m = month % 12
        self.monthly_profile[m] = np.where(
            mask,
            (1.0 - s_alpha) * self.monthly_profile[m] + s_alpha * deviation,
            self.monthly_profile[m],
        )
        self.monthly_counts[m] += mask

        self.n_samples += 1
        self.cov_sum += np.outer(residual, residual)
        self.cov_counts += np.outer(mask, mask)

    def expected_baseline(self, hour: int, month: int) -> NDArray[np.float64]:
        """Return expected baseline incorporating circadian + seasonal patterns."""
        h = hour % 24
        m = month % 12
        base = self.ema.copy()
        circadian_learned = self._profile_is_learned(self.circadian_counts[h])
        base = base + np.where(circadian_learned, 0.3 * self.circadian_profile[h], 0.0)
        monthly_learned = self._profile_is_learned(self.monthly_counts[m])
        return base + np.where(monthly_learned, 0.1 * self.monthly_profile[m], 0.0)

    def covariance_matrix(self) -> NDArray[np.float64]:
        """Prior-weighted covariance of the scored residual process.

        Entries are weighted by how many epochs contributed to *that pair* of
        channels, so a duty-cycled channel does not shrink the covariance of the
        channels that were present throughout.
        """
        if self.n_samples < COVARIANCE_WARMUP_SAMPLES:
            return self.covariance_prior.copy()
        denominator = self.covariance_prior_strength + self.cov_counts
        numerator = self.covariance_prior_strength * self.covariance_prior + self.cov_sum
        return numerator / denominator

    def mahalanobis_distance(
        self,
        observation: NDArray[np.float64],
        hour: int,
        month: int,
        observed: NDArray[np.bool_] | None = None,
    ) -> float:
        """Mahalanobis distance of the observed sub-vector from its baseline.

        Missing channels are marginalized out rather than imputed: the distance
        is computed on the observed sub-vector against the corresponding
        sub-matrix of the covariance. Its null distribution therefore has as
        many degrees of freedom as there are observed channels, which is what
        ``garland.thresholds`` needs to keep the alarm rate stable.
        """
        mask = self._observed_mask(observed)
        n_observed = int(mask.sum())
        if n_observed == 0:
            return 0.0
        baseline = self.expected_baseline(hour, month)
        diff = (observation - baseline)[mask]
        cov = self.covariance_matrix()[np.ix_(mask, mask)]
        cov_reg = cov + np.eye(n_observed) * 1e-6
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
