"""Regression tests for the calibrated cold-start covariance prior."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest
from scripts.coldstart_variance_check import measure

from garland.biometric_profiles import generate_profiles
from garland.biometric_synthesis import generate_observation
from garland.biometrics import BaselineTracker
from garland.channels import BODY_TEMPERATURE, CORE_VITALS, DEFAULT_CHANNEL_SET


@lru_cache(maxsize=1)
def _calibration_result() -> dict[str, object]:
    """Run the expensive calibration harness once for this test module."""
    return measure()


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
    result = _calibration_result()

    for row in result["rows"]:
        ratio = float(row["prior_to_measured_ratio"])
        assert 0.5 <= ratio <= 2.0, row


def test_cold_start_driver_share_moves_away_from_heart_rate() -> None:
    """Calibrated scaling reduces heart-rate dominance in benign cold alarms."""
    result = _calibration_result()
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


def test_population_mean_prior_starts_at_channel_resting_means() -> None:
    """The default mean prior is the channel registry's population mean."""
    tracker = BaselineTracker()
    np.testing.assert_allclose(tracker.ema, DEFAULT_CHANNEL_SET.resting_means)


def test_mean_prior_strength_grades_early_learning() -> None:
    """Larger pseudo-counts hold a device's initial mean longer."""
    channel = DEFAULT_CHANNEL_SET.index("heart_rate")
    observation = DEFAULT_CHANNEL_SET.resting_means.copy()
    observation[channel] += 20.0
    movements = []
    for strength in (1.0, 12.0, 48.0, 288.0):
        tracker = BaselineTracker(mean_prior_strength=strength)
        tracker.update(observation, hour=12, month=7)
        movements.append(float(tracker.ema[channel] - tracker.mean_prior[channel]))

    assert movements == sorted(movements, reverse=True)
    assert movements[0] - movements[-1] > 8.0


def test_zero_mean_source_preserves_legacy_initial_learning() -> None:
    """A zero prior with no pseudo-count reproduces the former EMA start."""
    tracker = BaselineTracker(
        mean_prior=DEFAULT_CHANNEL_SET.zeros(),
        mean_prior_strength=0.0,
    )
    observation = DEFAULT_CHANNEL_SET.resting_means
    tracker.update(observation, hour=12, month=7)
    expected = (1.0 - np.exp(-tracker.decay_lambda)) * observation
    np.testing.assert_allclose(tracker.ema, expected)


def test_population_prior_keeps_a_normal_observation_below_alarm_cut() -> None:
    """A normal fresh-device observation is not a cold-start alarm."""
    tracker = BaselineTracker()
    observation = tracker.ema + np.array([2.0, 1.0, 0.3, 0.01])
    assert tracker.mahalanobis_distance(observation, hour=12, month=7) < 3.5


def test_benign_day_covariance_stays_near_calibrated_variance() -> None:
    """Prior-mean learning prevents early covariance contamination."""
    result = _calibration_result()
    measured = {
        str(row["channel"]): float(row["benign_residual_variance"])
        for row in result["rows"]  # type: ignore[union-attr]
    }
    tracker = BaselineTracker()

    rng = np.random.default_rng(20260819)
    profile = generate_profiles(1, rng)[0]
    for step in range(288):
        hour_of_day = (step * 5 / 60.0) % 24
        observation = DEFAULT_CHANNEL_SET.clamp(
            generate_observation(
                profile,
                hour_of_day,
                day_of_year=196,
                rng=rng,
                activity_level=(
                    0.3 * max(0.0, math.sin(math.pi * (hour_of_day - 6) / 12))
                    if 6 <= hour_of_day <= 22
                    else 0.0
                )
                + rng.normal(0, 0.05),
            )
        )
        tracker.update(observation, hour=int(hour_of_day) % 24, month=7)

    for index, name in enumerate(DEFAULT_CHANNEL_SET.names):
        learned = float(tracker.covariance_matrix()[index, index])
        assert 0.25 * measured[name] <= learned <= 4.0 * measured[name]


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
