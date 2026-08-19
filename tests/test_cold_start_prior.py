"""Regression tests for the calibrated cold-start covariance prior."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scripts.coldstart_variance_check import measure

from garland.biometrics import BaselineTracker
from garland.channels import BODY_TEMPERATURE, CORE_VITALS, DEFAULT_CHANNEL_SET


def test_prior_variances_are_finite_positive_and_scale_with_channel_units() -> None:
    """The diagonal prior is positive and follows the measured scale ordering."""
    variances = DEFAULT_CHANNEL_SET.prior_variances

    assert np.all(np.isfinite(variances))
    assert np.all(variances > 0.0)
    assert (
        variances[DEFAULT_CHANNEL_SET.index("heart_rate")]
        > variances[DEFAULT_CHANNEL_SET.index("hrv_rmssd")]
    )
    assert (
        variances[DEFAULT_CHANNEL_SET.index("hrv_rmssd")]
        > variances[DEFAULT_CHANNEL_SET.index("respiratory_rate")]
    )
    assert (
        variances[DEFAULT_CHANNEL_SET.index("respiratory_rate")]
        > variances[DEFAULT_CHANNEL_SET.index("body_temperature")]
    )


def test_committed_priors_track_mature_benign_residual_variance() -> None:
    """Each prior stays within 2x of the reproducible calibration measurement.

    A factor-of-two band allows for sampling noise and deliberate physiology
    model evolution while still failing loudly if a channel's scale changes.
    """
    result = measure()

    for row in result["rows"]:
        ratio = float(row["prior_to_measured_ratio"])
        assert 0.5 <= ratio <= 2.0, row


def test_cold_start_driver_share_moves_away_from_heart_rate() -> None:
    """Calibrated scaling reduces heart-rate dominance in benign cold alarms."""
    result = measure()
    flat_share = float(result["flat_prior_driver_shares"]["heart_rate"])
    calibrated_share = float(result["cold_start_driver_shares"]["heart_rate"])

    assert flat_share > 0.5
    assert calibrated_share <= flat_share - 0.10


def _temperature_tracker() -> BaselineTracker:
    tracker = BaselineTracker()
    tracker.ema = np.array(
        [channel.resting_mean for channel in DEFAULT_CHANNEL_SET.channels],
        dtype=np.float64,
    )
    return tracker


def test_cold_temperature_threshold_is_many_prior_standard_deviations() -> None:
    """A threshold-sized fever is visible against a fresh covariance prior."""
    tracker = _temperature_tracker()
    temperature = DEFAULT_CHANNEL_SET.index(BODY_TEMPERATURE.name)
    observation = tracker.ema.copy()
    observation[temperature] += BODY_TEMPERATURE.deviation_threshold
    observed = np.zeros(len(CORE_VITALS), dtype=np.bool_)
    observed[temperature] = True

    distance = tracker.mahalanobis_distance(observation, 12, 7, observed)
    prior_sd = math.sqrt(BODY_TEMPERATURE.prior_variance)

    expected = BODY_TEMPERATURE.deviation_threshold / prior_sd
    assert distance == pytest.approx(expected, rel=1e-4)
    assert distance > 4.0


def test_temperature_sensitivity_increases_and_crosses_alarm_cut() -> None:
    """Increasing fever excursions produce increasing Mahalanobis distances."""
    tracker = _temperature_tracker()
    temperature = DEFAULT_CHANNEL_SET.index(BODY_TEMPERATURE.name)
    observed = np.zeros(len(CORE_VITALS), dtype=np.bool_)
    observed[temperature] = True
    distances = []
    for excursion in (0.2, 0.8, 1.6):
        observation = tracker.ema.copy()
        observation[temperature] += excursion
        distances.append(tracker.mahalanobis_distance(observation, 12, 7, observed))

    assert distances == sorted(distances)
    assert distances[0] < 3.5
    assert distances[1] > 3.5
