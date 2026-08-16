"""Tests for degrees-of-freedom threshold calibration and masked scoring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from garland.agents import ANOMALY_THRESHOLD, CitizenAgent
from garland.biometric_profiles import build_profile
from garland.biometric_synthesis import generate_observation_custom
from garland.biometrics import BaselineTracker
from garland.channels import CORE_VITALS
from garland.detection import SequentialDetector
from garland.thresholds import (
    REFERENCE_DOF,
    calibrated_threshold,
    chi_mean,
    chi_square_survival,
    chi_square_upper_quantile,
    per_epoch_false_positive_rate,
    reference_value_for_dof,
    threshold_for_dof,
)
from tests.test_channels import WIDE_SET


class TestChiSquareTail:
    def test_survival_matches_closed_form_for_even_dof(self):
        """Even-dof chi-square tails have an exact elementary form to check against."""
        for dof in (2, 4, 6, 10):
            for statistic in (0.5, 2.0, 7.5, 12.25, 30.0):
                half = 0.5 * statistic
                terms = sum(half**i / math.factorial(i) for i in range(dof // 2))
                expected = float(np.exp(-half) * terms)
                assert chi_square_survival(statistic, dof) == pytest.approx(expected, rel=1e-10)

    def test_survival_is_a_decreasing_probability(self):
        for dof in (1, 3, 4, 14):
            values = [chi_square_survival(x, dof) for x in (0.0, 1.0, 5.0, 20.0, 60.0)]
            assert values[0] == pytest.approx(1.0)
            assert all(0.0 <= value <= 1.0 for value in values)
            assert all(later < earlier for earlier, later in zip(values, values[1:], strict=False))

    def test_quantile_inverts_survival(self):
        """A few tail probabilities give back statistics with those exact tails."""
        for dof in (1, 4, 9, 14):
            for tail in (0.2, 0.05, 0.01, 1e-4):
                statistic = chi_square_upper_quantile(tail, dof)
                assert chi_square_survival(statistic, dof) == pytest.approx(tail, rel=1e-6)

    def test_degenerate_arguments_rejected(self):
        with pytest.raises(ValueError, match="dof must be at least 1"):
            chi_square_survival(1.0, 0)
        for tail in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="strictly between"):
                chi_square_upper_quantile(tail, 4)


class TestThresholdCalibration:
    def test_default_threshold_rate_is_preserved_across_widths(self):
        """Widening the vector holds the per-epoch alarm rate, not the distance."""
        reference_rate = per_epoch_false_positive_rate(ANOMALY_THRESHOLD, REFERENCE_DOF)
        for dof in (1, 2, 4, 6, 9, 14, 24):
            threshold = threshold_for_dof(ANOMALY_THRESHOLD, dof)
            assert per_epoch_false_positive_rate(threshold, dof) == pytest.approx(
                reference_rate, rel=1e-6
            )

    def test_reference_width_is_left_exactly_alone(self):
        """The historical four-channel cut must survive bit-for-bit."""
        assert threshold_for_dof(ANOMALY_THRESHOLD, REFERENCE_DOF) == ANOMALY_THRESHOLD
        assert threshold_for_dof(4.75, REFERENCE_DOF) == pytest.approx(4.75, abs=0.0)

    def test_wider_sets_need_larger_cuts(self):
        thresholds = [threshold_for_dof(ANOMALY_THRESHOLD, dof) for dof in (2, 4, 8, 16)]
        assert all(
            later > earlier for earlier, later in zip(thresholds, thresholds[1:], strict=False)
        )
        assert thresholds[1] == pytest.approx(ANOMALY_THRESHOLD)

    def test_uncalibrated_fixed_cut_would_inflate_the_alarm_rate(self):
        """Documents the failure this module exists to prevent."""
        four = per_epoch_false_positive_rate(ANOMALY_THRESHOLD, 4)
        fourteen = per_epoch_false_positive_rate(ANOMALY_THRESHOLD, 14)
        assert fourteen > 5.0 * four

    def test_stricter_target_rate_raises_the_cut(self):
        cuts = [calibrated_threshold(6, rate) for rate in (0.1, 0.01, 0.001)]
        assert all(later > earlier for earlier, later in zip(cuts, cuts[1:], strict=False))

    def test_nonpositive_reference_threshold_passes_through(self):
        """A cut that flags everything has no rate to preserve (used by tests/configs)."""
        assert threshold_for_dof(-1.0, 9) == pytest.approx(-1.0, abs=0.0)


class TestSequentialSlackCalibration:
    def test_chi_mean_tracks_sqrt_dof(self):
        """The resting distance grows like sqrt(dof), which is why slack must scale."""
        for dof in (1, 4, 9, 16, 25):
            assert chi_mean(dof) == pytest.approx(math.sqrt(dof), rel=0.35)
        means = [chi_mean(dof) for dof in (1, 4, 9, 16)]
        assert all(later > earlier for earlier, later in zip(means, means[1:], strict=False))

    def test_reference_width_slack_unchanged(self):
        assert reference_value_for_dof(2.0, REFERENCE_DOF) == pytest.approx(2.0, abs=0.0)

    def test_slack_stays_above_the_null_mean_at_every_width(self):
        """Otherwise the CUSUM statistic ramps to an alarm on quiet data."""
        slack_at_reference = 2.0
        assert slack_at_reference > chi_mean(REFERENCE_DOF)
        for dof in (1, 2, 6, 14, 30):
            assert reference_value_for_dof(slack_at_reference, dof) > chi_mean(dof)

    def test_quiet_wide_detector_does_not_ramp(self):
        detector = SequentialDetector(channel_set=WIDE_SET)
        residual = WIDE_SET.zeros()
        for _ in range(200):
            detector.update(chi_mean(len(WIDE_SET)), residual)
        assert not detector.alarm_active
        assert detector.statistic == pytest.approx(0.0)

    def test_sustained_wide_excursion_still_alarms(self):
        detector = SequentialDetector(channel_set=WIDE_SET)
        residual = WIDE_SET.delta({"body_temperature": 1.5})
        alarmed = any(detector.update(3.0 * chi_mean(len(WIDE_SET)), residual) for _ in range(200))
        assert alarmed
        assert detector.alarm_active


class TestMaskedScoring:
    @staticmethod
    def _warm_tracker(channel_set, seed=101, steps=120):
        tracker = BaselineTracker(channel_set=channel_set)
        profile = build_profile(channel_set=channel_set)
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            tracker.update(generate_observation_custom(profile, 12.0, 180, rng), 12, 6)
        return tracker, profile, rng

    def test_full_mask_matches_unmasked_scoring(self):
        tracker, profile, rng = self._warm_tracker(WIDE_SET)
        obs = generate_observation_custom(profile, 12.0, 180, rng)
        full = np.ones(len(WIDE_SET), dtype=np.bool_)
        assert tracker.mahalanobis_distance(obs, 12, 6, full) == pytest.approx(
            tracker.mahalanobis_distance(obs, 12, 6)
        )

    def test_missing_channels_are_marginalized_not_imputed(self):
        """A masked channel's deviation must not reach the score at all."""
        tracker, profile, rng = self._warm_tracker(WIDE_SET)
        obs = generate_observation_custom(profile, 12.0, 180, rng)
        mask = np.ones(len(WIDE_SET), dtype=np.bool_)
        mask[WIDE_SET.index("cough_rate")] = False
        clean = tracker.mahalanobis_distance(obs, 12, 6, mask)
        spiked = obs + WIDE_SET.delta({"cough_rate": 50.0})
        assert tracker.mahalanobis_distance(spiked, 12, 6, mask) == pytest.approx(clean)
        assert tracker.mahalanobis_distance(spiked, 12, 6) > clean

    def test_score_grades_with_the_number_of_observed_channels(self):
        """Dropping channels from a uniform anomaly monotonically lowers the score."""
        tracker, profile, rng = self._warm_tracker(CORE_VITALS)
        obs = generate_observation_custom(profile, 12.0, 180, rng)
        anomalous = obs + CORE_VITALS.delta(
            {
                "heart_rate": 18.0,
                "hrv_rmssd": -14.0,
                "respiratory_rate": 5.0,
                "body_temperature": 1.2,
            }
        )
        scores = []
        for n_observed in (1, 2, 3, 4):
            mask = np.zeros(len(CORE_VITALS), dtype=np.bool_)
            mask[:n_observed] = True
            scores.append(tracker.mahalanobis_distance(anomalous, 12, 6, mask))
        assert all(np.isfinite(scores))
        assert all(later > earlier for earlier, later in zip(scores, scores[1:], strict=False))

    def test_empty_mask_scores_zero(self):
        tracker, profile, rng = self._warm_tracker(CORE_VITALS)
        obs = generate_observation_custom(profile, 12.0, 180, rng)
        empty = np.zeros(len(CORE_VITALS), dtype=np.bool_)
        assert tracker.mahalanobis_distance(obs, 12, 6, empty) == pytest.approx(0.0)

    def test_mask_width_is_validated(self):
        tracker = BaselineTracker(channel_set=CORE_VITALS)
        with pytest.raises(ValueError, match="observed mask has"):
            tracker.mahalanobis_distance(CORE_VITALS.zeros(), 12, 6, np.ones(3, dtype=np.bool_))

    def test_masked_updates_leave_absent_channels_unlearned(self):
        """A never-reported channel keeps its prior baseline and prior variance."""
        tracker = BaselineTracker(channel_set=WIDE_SET)
        profile = build_profile(channel_set=WIDE_SET)
        rng = np.random.default_rng(103)
        absent = WIDE_SET.index("cough_rate")
        mask = np.ones(len(WIDE_SET), dtype=np.bool_)
        mask[absent] = False
        for _ in range(60):
            tracker.update(generate_observation_custom(profile, 12.0, 180, rng), 12, 6, mask)
        assert tracker.ema[absent] == pytest.approx(0.0)
        assert tracker.circadian_profile[12, absent] == pytest.approx(0.0)
        assert tracker.covariance_matrix()[absent, absent] == pytest.approx(
            WIDE_SET.prior_variances[absent]
        )
        present = WIDE_SET.index("heart_rate")
        assert tracker.ema[present] > 0.0
        learned_variance = tracker.covariance_matrix()[present, present]
        assert np.isfinite(learned_variance)
        assert learned_variance != pytest.approx(WIDE_SET.prior_variances[present])

    def test_duty_cycled_channel_does_not_dilute_present_channels(self):
        """Pairwise counts keep a sparse channel from shrinking others' variance."""
        profile = build_profile(channel_set=WIDE_SET)
        sparse_index = WIDE_SET.index("cough_rate")
        present_index = WIDE_SET.index("heart_rate")
        variances = []
        for report_every in (1, 10):
            tracker = BaselineTracker(channel_set=WIDE_SET)
            rng = np.random.default_rng(107)
            for step in range(80):
                mask = np.ones(len(WIDE_SET), dtype=np.bool_)
                mask[sparse_index] = step % report_every == 0
                tracker.update(generate_observation_custom(profile, 12.0, 180, rng), 12, 6, mask)
            variances.append(tracker.covariance_matrix()[present_index, present_index])
        assert variances[1] == pytest.approx(variances[0], rel=0.35)


class TestAgentThresholdIntegration:
    @staticmethod
    def _agent(channel_set, detector_mode="instant"):
        return CitizenAgent(
            idx=1,
            has_wearable=True,
            profile=build_profile(channel_set=channel_set),
            detector_mode=detector_mode,
        )

    def test_agent_scores_only_the_reported_channels(self):
        agent = self._agent(WIDE_SET)
        rng = np.random.default_rng(211)
        mask = np.ones(len(WIDE_SET), dtype=np.bool_)
        mask[WIDE_SET.index("cough_rate")] = False
        for _ in range(30):
            agent.observe_and_detect(12, 6, 180, 12.0, rng, cell_id=0, observed_channels=mask)
        assert agent.baseline.ema[WIDE_SET.index("cough_rate")] == pytest.approx(0.0)
        assert agent.baseline.ema[WIDE_SET.index("heart_rate")] > 0.0

    def test_all_missing_epoch_reports_nothing(self):
        agent = self._agent(CORE_VITALS)
        rng = np.random.default_rng(213)
        empty = np.zeros(len(CORE_VITALS), dtype=np.bool_)
        token = agent.observe_and_detect(12, 6, 180, 12.0, rng, cell_id=0, observed_channels=empty)
        assert token is None
        assert agent.baseline.n_samples == 0
        assert not agent.anomaly_active

    def test_wider_fleet_does_not_alarm_more_often_at_rest(self):
        """Quiet wide-set agents must not out-alarm quiet default-set agents."""
        alarm_counts = []
        for channel_set in (CORE_VITALS, WIDE_SET):
            alarms = 0
            for seed in range(12):
                agent = self._agent(channel_set)
                rng = np.random.default_rng(300 + seed)
                for step in range(60):
                    token = agent.observe_and_detect(
                        12,
                        6,
                        180,
                        12.0,
                        rng,
                        cell_id=0,
                        suppress_token_emission=step < 20,
                    )
                    alarms += token is not None
            alarm_counts.append(alarms)
        assert alarm_counts[1] <= alarm_counts[0] + 3
