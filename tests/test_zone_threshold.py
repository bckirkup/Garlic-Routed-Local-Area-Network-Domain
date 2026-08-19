"""Tests for the fleet zone-trigger recalibration.

The mechanism under test converts a measured distribution of quiet zone-window
token counts into the trigger count that meets a configured false-trigger rate,
so the tests are mostly graded sweeps over that distribution and over the target
rate, plus the bounds the count is never allowed to leave.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.privacy import AggregatorState, AnomalyType, EncryptedToken, PrivacyConfig
from garland.zone_threshold import (
    MAX_TRACKED_COUNT,
    ZoneThresholdCalibrationConfig,
    ZoneThresholdCalibrator,
    zone_threshold_calibrator_for,
)


def _calibrator(**overrides: object) -> ZoneThresholdCalibrator:
    """Build a collecting calibrator with a small sample requirement."""
    settings: dict[str, object] = {"enabled": True, "min_samples": 100}
    settings.update(overrides)
    calibrator = zone_threshold_calibrator_for(
        ZoneThresholdCalibrationConfig(**settings),  # type: ignore[arg-type]
        configured_threshold=5,
    )
    calibrator.collecting = True
    return calibrator


def _feed_poisson(calibrator: ZoneThresholdCalibrator, mean: float, n: int = 20_000) -> None:
    """Fold ``n`` Poisson window counts with the given mean into a calibrator."""
    rng = np.random.default_rng(20260816)
    for count in rng.poisson(mean, size=n):
        calibrator.observe(int(count))


def _learned_threshold(mean: float, **overrides: object) -> int:
    """Threshold learned from a Poisson quiet-window distribution."""
    calibrator = _calibrator(**overrides)
    _feed_poisson(calibrator, mean)
    calibrator.freeze()
    return calibrator.threshold


class TestConfigValidation:
    """The configuration refuses operating points it cannot honour."""

    def test_window_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="end_step must exceed start_step"):
            ZoneThresholdCalibrationConfig(start_step=500, end_step=500)

    def test_start_step_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="start_step must be non-negative"):
            ZoneThresholdCalibrationConfig(start_step=-1)

    @pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
    def test_false_trigger_rate_must_be_a_probability(self, rate: float) -> None:
        with pytest.raises(ValueError, match="false_trigger_rate"):
            ZoneThresholdCalibrationConfig(false_trigger_rate=rate)

    def test_single_device_broadcasts_are_not_configurable(self) -> None:
        with pytest.raises(ValueError, match="minimum_threshold must be at least 2"):
            ZoneThresholdCalibrationConfig(minimum_threshold=1)

    def test_ceiling_must_not_undercut_floor(self) -> None:
        with pytest.raises(ValueError, match="maximum_threshold"):
            ZoneThresholdCalibrationConfig(minimum_threshold=8, maximum_threshold=4)

    def test_min_samples_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="min_samples must be at least 1"):
            ZoneThresholdCalibrationConfig(min_samples=0)

    def test_configured_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="configured_threshold must be positive"):
            ZoneThresholdCalibrator(configured_threshold=0)


class TestSensitivity:
    """Graded response to the measured token rate and the target rate."""

    def test_busier_quiet_traffic_demands_more_tokens(self) -> None:
        means = [0.2, 1.0, 3.0, 8.0]
        thresholds = [_learned_threshold(mean) for mean in means]

        assert thresholds == sorted(thresholds)
        assert thresholds[-1] - thresholds[0] >= 8, (
            f"trigger count barely tracks the token rate: {thresholds}"
        )

    def test_looser_target_rate_lowers_the_trigger_count(self) -> None:
        rates = [0.0005, 0.005, 0.05, 0.2]
        thresholds = [_learned_threshold(3.0, false_trigger_rate=rate) for rate in rates]

        assert thresholds == sorted(thresholds, reverse=True)
        assert thresholds[0] - thresholds[-1] >= 3, (
            f"target false-trigger rate looks like a dead knob: {thresholds}"
        )

    def test_learned_count_holds_the_requested_tail(self) -> None:
        """The count is the smallest one meeting the rate, so the tail brackets it."""
        rate = 0.01
        mean = 4.0
        calibrator = _calibrator(false_trigger_rate=rate)
        _feed_poisson(calibrator, mean)
        calibrator.freeze()

        rng = np.random.default_rng(4242)
        counts = rng.poisson(mean, size=40_000)
        achieved = float(np.mean(counts >= calibrator.threshold))
        one_lower = float(np.mean(counts >= calibrator.threshold - 1))

        assert achieved <= rate * 1.5
        assert one_lower > rate

    def test_calibrated_count_is_stricter_than_a_constant_at_high_density(self) -> None:
        """A dense fleet needs more than the shipped constant, a sparse one less."""
        sparse = _learned_threshold(0.3)
        dense = _learned_threshold(10.0)

        assert sparse < 5 < dense

    def test_unrelated_window_bounds_do_not_move_the_count(self) -> None:
        baseline = _learned_threshold(3.0)
        shifted = _learned_threshold(3.0, start_step=12, end_step=2000)

        assert shifted == baseline


class TestBoundsAndInvariants:
    """The learned count stays inside its declared range."""

    @pytest.mark.parametrize("mean", [0.0, 0.5, 5.0, 40.0, 500.0])
    def test_threshold_stays_within_configured_bounds(self, mean: float) -> None:
        calibrator = _calibrator(minimum_threshold=3, maximum_threshold=17)
        _feed_poisson(calibrator, mean, n=5_000)
        calibrator.freeze()

        assert 3 <= calibrator.threshold <= 17

    def test_silent_fleet_falls_back_to_the_floor(self) -> None:
        calibrator = _calibrator(minimum_threshold=4)
        for _ in range(1_000):
            calibrator.observe(0)
        calibrator.freeze()

        assert calibrator.threshold == 4

    def test_saturating_traffic_clamps_to_the_ceiling(self) -> None:
        calibrator = _calibrator(maximum_threshold=12)
        for _ in range(1_000):
            calibrator.observe(MAX_TRACKED_COUNT * 4)
        calibrator.freeze()

        assert calibrator.threshold == 12

    def test_thin_evidence_keeps_the_configured_count(self) -> None:
        calibrator = _calibrator(min_samples=5_000)
        _feed_poisson(calibrator, 6.0, n=100)
        calibrator.freeze()

        assert calibrator.threshold == calibrator.configured_threshold == 5

    def test_negative_counts_are_ignored(self) -> None:
        calibrator = _calibrator()
        calibrator.observe(-3)

        assert calibrator.samples == 0

    def test_counts_outside_the_window_are_ignored(self) -> None:
        calibrator = _calibrator()
        calibrator.collecting = False
        calibrator.observe(4)

        assert calibrator.samples == 0

    def test_frozen_calibrator_stops_learning(self) -> None:
        calibrator = _calibrator()
        _feed_poisson(calibrator, 1.0, n=2_000)
        calibrator.freeze()
        learned = calibrator.threshold
        folded = calibrator.samples

        for _ in range(5_000):
            calibrator.observe(MAX_TRACKED_COUNT)
        calibrator.freeze()

        assert calibrator.samples == folded
        assert calibrator.threshold == learned

    def test_calibration_is_deterministic_for_one_count_sequence(self) -> None:
        first = _learned_threshold(2.5)
        second = _learned_threshold(2.5)

        assert first == second


class TestWindow:
    """The calibration window opens, holds and closes on simulation steps."""

    def test_window_opens_and_closes_on_configured_steps(self) -> None:
        calibrator = zone_threshold_calibrator_for(
            ZoneThresholdCalibrationConfig(enabled=True, start_step=100, end_step=200),
            configured_threshold=5,
        )

        calibrator.advance(50)
        assert not calibrator.collecting
        calibrator.advance(100)
        assert calibrator.collecting
        calibrator.advance(199)
        assert calibrator.collecting
        calibrator.advance(200)
        assert calibrator.frozen
        assert not calibrator.collecting

    def test_summary_reports_the_operating_point(self) -> None:
        calibrator = _calibrator(false_trigger_rate=0.02)
        _feed_poisson(calibrator, 2.0, n=1_000)
        calibrator.freeze()
        summary = calibrator.summary()

        assert summary["frozen"] is True
        assert summary["configured_threshold_m"] == 5
        assert summary["threshold_m"] == calibrator.threshold
        assert summary["calibration_windows"] == 1_000
        assert summary["target_false_trigger_rate"] == pytest.approx(0.02)


class TestAggregatorIntegration:
    """The aggregator's window counts drive, and then obey, the learned count."""

    @staticmethod
    def _state_with_tokens(count: int, time_bin: int = 3) -> AggregatorState:
        state = AggregatorState()
        for index in range(count):
            state.receive_token(
                EncryptedToken(
                    zone_id=7,
                    anomaly_type=AnomalyType.RESPIRATORY,
                    timestamp_bin=time_bin,
                    agent_id_hash=index,
                    is_dummy=False,
                )
            )
        return state

    def test_configured_count_applies_before_freezing(self) -> None:
        config = PrivacyConfig(threshold_m=5)
        calibrator = _calibrator()
        state = self._state_with_tokens(4)

        assert state.check_thresholds(3, config, calibrator) == []
        assert calibrator.samples == 1

        state.receive_token(
            EncryptedToken(
                zone_id=7,
                anomaly_type=AnomalyType.RESPIRATORY,
                timestamp_bin=3,
                agent_id_hash=99,
                is_dummy=False,
            )
        )
        assert state.check_thresholds(3, config, calibrator) == [(7, AnomalyType.RESPIRATORY)]

    def test_frozen_count_replaces_the_configured_one(self) -> None:
        config = PrivacyConfig(threshold_m=5)
        calibrator = _calibrator(minimum_threshold=3, maximum_threshold=3)
        _feed_poisson(calibrator, 6.0, n=1_000)
        calibrator.freeze()
        state = self._state_with_tokens(3)

        assert calibrator.threshold == 3
        assert state.check_thresholds(3, config, calibrator) == [(7, AnomalyType.RESPIRATORY)]

    def test_absent_calibrator_leaves_thresholding_unchanged(self) -> None:
        """Negative control: the default path is the configured constant."""
        config = PrivacyConfig(threshold_m=5)

        assert self._state_with_tokens(4).check_thresholds(3, config) == []
        assert self._state_with_tokens(5).check_thresholds(3, config) == [
            (7, AnomalyType.RESPIRATORY)
        ]

    def test_frozen_calibrator_does_not_learn_from_hazard_windows(self) -> None:
        config = PrivacyConfig(threshold_m=5)
        calibrator = _calibrator()
        _feed_poisson(calibrator, 1.0, n=1_000)
        calibrator.freeze()
        learned = calibrator.threshold
        state = self._state_with_tokens(30)

        state.check_thresholds(3, config, calibrator)

        assert calibrator.threshold == learned
