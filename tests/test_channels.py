"""Tests for the channel registry and variable-width observation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from garland.agents import CitizenAgent
from garland.biometric_profiles import BiometricProfile, build_profile, generate_profiles
from garland.biometric_synthesis import generate_observation_custom
from garland.biometrics import BaselineTracker
from garland.channels import (
    CORE_VITALS,
    DEFAULT_CHANNEL_SET,
    Channel,
    ChannelSet,
    ChannelSystem,
)
from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.detection import SequentialDetector
from garland.hazards import SEIRConfig, SEIREngine, SEIRState, plume_biometric_perturbation
from garland.openwearables import observation_to_records
from garland.privacy import AnomalyType, classify_anomaly

# A wider fleet: one extra channel per system plus a channel with no Open
# Wearables equivalent, exercising width-generality without asserting any
# particular future modality's calibration.
SYSTOLIC_BP = Channel(
    name="systolic_bp",
    unit="mmHg",
    system=ChannelSystem.CARDIAC,
    resting_mean=118.0,
    resting_sd=10.0,
    resting_min=90.0,
    resting_max=160.0,
    noise_sd=3.0,
    deviation_threshold=12.0,
    circadian_amp_min=2.0,
    circadian_amp_max=6.0,
    openwearables_type="blood_pressure_systolic",
)

COUGH_RATE = Channel(
    name="cough_rate",
    unit="events/hour",
    system=ChannelSystem.RESPIRATORY,
    resting_mean=0.4,
    resting_sd=0.3,
    resting_min=0.0,
    resting_max=4.0,
    noise_sd=0.1,
    deviation_threshold=2.0,
    floor=0.0,
)

WIDE_SET = CORE_VITALS.with_channels(SYSTOLIC_BP, COUGH_RATE)


class TestChannelSet:
    def test_core_set_is_the_default_layout(self):
        assert DEFAULT_CHANNEL_SET is CORE_VITALS
        assert len(CORE_VITALS) == 4
        assert CORE_VITALS.names == (
            "heart_rate",
            "hrv_rmssd",
            "respiratory_rate",
            "body_temperature",
        )

    def test_named_lookup_matches_vector_order(self):
        for position, name in enumerate(WIDE_SET.names):
            assert WIDE_SET.index(name) == position
            assert WIDE_SET.get(name).name == name
            assert WIDE_SET.has(name)
        assert not WIDE_SET.has("eeg_delta_power")

    def test_widening_preserves_the_prefix_layout(self):
        assert WIDE_SET.names[: len(CORE_VITALS)] == CORE_VITALS.names
        assert len(WIDE_SET) == len(CORE_VITALS) + 2

    def test_unknown_channel_rejected(self):
        with pytest.raises(KeyError, match="eeg_delta_power"):
            CORE_VITALS.index("eeg_delta_power")
        with pytest.raises(KeyError, match="systolic_bp"):
            CORE_VITALS.delta({"systolic_bp": 5.0})

    def test_malformed_sets_rejected(self):
        with pytest.raises(ValueError, match="at least one channel"):
            ChannelSet(())
        with pytest.raises(ValueError, match="duplicate channel"):
            ChannelSet((SYSTOLIC_BP, SYSTOLIC_BP))

    def test_delta_places_named_values_only(self):
        delta = WIDE_SET.delta({"cough_rate": 3.0, "heart_rate": -2.0})
        assert delta[WIDE_SET.index("cough_rate")] == pytest.approx(3.0)
        assert delta[WIDE_SET.index("heart_rate")] == pytest.approx(-2.0)
        assert delta.shape == (len(WIDE_SET),)
        assert np.count_nonzero(delta) == 2

    def test_system_indices_group_channels(self):
        cardiac = WIDE_SET.system_indices(ChannelSystem.CARDIAC)
        assert set(cardiac) == {
            WIDE_SET.index("heart_rate"),
            WIDE_SET.index("hrv_rmssd"),
            WIDE_SET.index("systolic_bp"),
        }
        assert WIDE_SET.system_indices(ChannelSystem.THERMAL) == (
            WIDE_SET.index("body_temperature"),
        )

    def test_parameter_vectors_align_with_channel_order(self):
        thresholds = WIDE_SET.deviation_thresholds
        assert thresholds.shape == (len(WIDE_SET),)
        assert thresholds[WIDE_SET.index("cough_rate")] == pytest.approx(
            COUGH_RATE.deviation_threshold
        )
        assert np.all(WIDE_SET.prior_variances > 0.0)
        assert np.all(np.isfinite(WIDE_SET.prior_variances))


class TestProfileWidth:
    def test_profiles_follow_the_channel_set(self):
        rng = np.random.default_rng(7)
        for channel_set in (CORE_VITALS, WIDE_SET):
            profiles = generate_profiles(5, rng, channel_set)
            for profile in profiles:
                assert profile.resting.shape == (len(channel_set),)
                assert profile.circadian_amp.shape == (len(channel_set),)
                assert profile.channel_set is channel_set

    def test_generated_resting_values_stay_in_channel_bounds(self):
        profiles = generate_profiles(200, np.random.default_rng(11), WIDE_SET)
        resting = np.array([profile.resting for profile in profiles])
        for position, channel in enumerate(WIDE_SET.channels):
            column = resting[:, position]
            assert np.all(np.isfinite(column))
            assert column.min() >= channel.resting_min
            assert column.max() <= channel.resting_max

    def test_mismatched_vector_width_rejected(self):
        with pytest.raises(ValueError, match="profile vectors must have"):
            BiometricProfile(
                resting=np.zeros(3),
                circadian_amp=np.zeros(3),
                channel_set=CORE_VITALS,
            )

    def test_build_profile_defaults_to_channel_means(self):
        profile = build_profile(resting={"heart_rate": 55.0}, channel_set=WIDE_SET)
        assert profile.resting_value("heart_rate") == pytest.approx(55.0)
        assert profile.resting_value("systolic_bp") == pytest.approx(SYSTOLIC_BP.resting_mean)
        assert profile.circadian_amplitude("cough_rate") == pytest.approx(0.0)
        with pytest.raises(KeyError):
            build_profile(resting={"eeg_delta_power": 1.0})


class TestSynthesisWidth:
    def test_observation_width_and_finiteness(self):
        rng = np.random.default_rng(3)
        for channel_set in (CORE_VITALS, WIDE_SET):
            profile = build_profile(channel_set=channel_set)
            obs = generate_observation_custom(profile, 14.0, 180, rng)
            assert obs.shape == (len(channel_set),)
            assert np.all(np.isfinite(obs))

    def test_channel_floor_is_enforced_under_load(self):
        profile = build_profile(resting={"cough_rate": 0.0}, channel_set=WIDE_SET)
        rng = np.random.default_rng(5)
        position = WIDE_SET.index("cough_rate")
        values = [
            generate_observation_custom(profile, 3.0, 10, rng, activity_level=1.0)[position]
            for _ in range(50)
        ]
        assert min(values) >= 0.0

    def test_resting_level_grades_the_synthesized_channel(self):
        """A few different resting levels give a few different observed means."""
        position = WIDE_SET.index("systolic_bp")
        means = []
        for resting in (100.0, 120.0, 140.0):
            profile = build_profile(resting={"systolic_bp": resting}, channel_set=WIDE_SET)
            rng = np.random.default_rng(19)
            samples = [
                generate_observation_custom(profile, 12.0, 180, rng)[position] for _ in range(40)
            ]
            means.append(float(np.mean(samples)))
        assert means[0] < means[1] < means[2]
        assert all(abs(mean - resting) < 5.0 for mean, resting in zip(means, (100.0, 120.0, 140.0)))

    def test_added_channels_do_not_disturb_core_channels(self):
        """The default four channels keep their exact draws when a set widens."""
        core_profile = build_profile(channel_set=CORE_VITALS)
        wide_profile = build_profile(channel_set=WIDE_SET)
        core = generate_observation_custom(core_profile, 9.0, 45, np.random.default_rng(23))
        wide = generate_observation_custom(wide_profile, 9.0, 45, np.random.default_rng(23))
        np.testing.assert_allclose(core, wide[: len(CORE_VITALS)])


class TestBaselineAndDetectorWidth:
    def test_baseline_state_matches_channel_width(self):
        tracker = BaselineTracker(channel_set=WIDE_SET)
        width = len(WIDE_SET)
        assert tracker.ema.shape == (width,)
        assert tracker.circadian_profile.shape == (24, width)
        assert tracker.monthly_profile.shape == (12, width)
        assert tracker.covariance_matrix().shape == (width, width)

    def test_wide_baseline_scores_stay_finite_and_ordered(self):
        tracker = BaselineTracker(channel_set=WIDE_SET)
        profile = build_profile(channel_set=WIDE_SET)
        rng = np.random.default_rng(29)
        for _ in range(100):
            tracker.update(generate_observation_custom(profile, 12.0, 180, rng), 12, 6)
        normal = generate_observation_custom(profile, 12.0, 180, rng)
        anomalous = normal + WIDE_SET.delta({"systolic_bp": 60.0, "cough_rate": 25.0})
        normal_distance = tracker.mahalanobis_distance(normal, 12, 6)
        anomalous_distance = tracker.mahalanobis_distance(anomalous, 12, 6)
        assert np.isfinite(normal_distance)
        assert normal_distance >= 0.0
        assert anomalous_distance > normal_distance

    def test_detector_state_matches_channel_width(self):
        detector = SequentialDetector(channel_set=WIDE_SET)
        assert detector.residual_ewma.shape == (len(WIDE_SET),)
        default_detector = SequentialDetector()
        assert default_detector.residual_ewma.shape == (len(CORE_VITALS),)

    def test_agent_state_follows_its_profile_layout(self):
        """A wider profile widens the agent's own state, baseline, and detector."""
        agent = CitizenAgent(
            idx=1,
            has_wearable=True,
            profile=build_profile(channel_set=WIDE_SET),
            detector_mode="sequential",
        )
        assert agent.channel_set == WIDE_SET
        assert agent.last_observation.shape == (len(WIDE_SET),)
        assert agent.baseline.channel_set == WIDE_SET
        assert agent.baseline.ema.shape == (len(WIDE_SET),)
        assert agent.sequential_detector is not None
        assert agent.sequential_detector.residual_ewma.shape == (len(WIDE_SET),)

        default_agent = CitizenAgent(idx=2, has_wearable=True)
        assert default_agent.channel_set == CORE_VITALS
        assert default_agent.baseline.ema.shape == (len(CORE_VITALS),)


class TestPerturbationWidth:
    def test_hazard_signatures_match_channel_width(self):
        seir = SEIREngine(SEIRConfig(initial_infected=1))
        seir.initialize(8, np.random.default_rng(31))
        seir.states[0] = SEIRState.INFECTIOUS
        for channel_set in (CORE_VITALS, WIDE_SET):
            delta = seir.biometric_perturbation(0, 12, channel_set)
            plume = plume_biometric_perturbation(1.0, channel_set)
            assert delta.shape == (len(channel_set),)
            assert plume.shape == (len(channel_set),)
            assert np.all(np.isfinite(delta))
            assert np.all(np.isfinite(plume))

    def test_core_channel_signatures_are_unchanged_by_widening(self):
        seir = SEIREngine(SEIRConfig(initial_infected=1))
        seir.initialize(4, np.random.default_rng(37))
        seir.states[0] = SEIRState.INFECTIOUS
        core = seir.biometric_perturbation(0, 20, CORE_VITALS)
        wide = seir.biometric_perturbation(0, 20, WIDE_SET)
        assert np.count_nonzero(core) > 0
        np.testing.assert_allclose(core, wide[: len(CORE_VITALS)])
        np.testing.assert_allclose(wide[len(CORE_VITALS) :], 0.0)

    def test_confounder_deltas_match_channel_width(self):
        config = ConfoundersConfig(enabled=True, exercise_rate=1.0)
        engine = ConfounderEngine(
            12,
            config,
            np.random.default_rng(41),
            channel_set=WIDE_SET,
        )
        step = engine.step(0, 8.0, np.ones(12, dtype=bool))
        contributions = [
            contribution
            for agent_contributions in step.contributions.values()
            for contribution in agent_contributions
        ]
        assert contributions
        for contribution in contributions:
            assert contribution.delta.shape == (len(WIDE_SET),)
            assert np.all(np.isfinite(contribution.delta))


class TestClassificationWidth:
    def test_core_classification_semantics_preserved(self):
        baseline = np.array([70.0, 40.0, 15.0, 36.8])
        assert classify_anomaly(baseline.copy(), baseline) is None
        assert (
            classify_anomaly(baseline + np.array([0.0, 0.0, 6.0, 0.1]), baseline)
            is AnomalyType.RESPIRATORY
        )
        assert (
            classify_anomaly(baseline + np.array([0.0, 0.0, 0.0, 1.2]), baseline)
            is AnomalyType.FEBRILE
        )
        assert (
            classify_anomaly(baseline + np.array([15.0, 0.0, 0.0, 0.0]), baseline)
            is AnomalyType.CARDIAC
        )
        assert (
            classify_anomaly(baseline + np.array([15.0, -12.0, 0.0, 1.2]), baseline)
            is AnomalyType.MULTI_SYSTEM
        )

    def test_added_cardiac_channel_classifies_as_cardiac(self):
        baseline = np.array([70.0, 40.0, 15.0, 36.8, 118.0, 0.4])
        observation = baseline + WIDE_SET.delta({"systolic_bp": 20.0})
        assert classify_anomaly(observation, baseline, WIDE_SET) is AnomalyType.CARDIAC

    def test_added_respiratory_channel_classifies_as_respiratory(self):
        baseline = np.array([70.0, 40.0, 15.0, 36.8, 118.0, 0.4])
        observation = baseline + WIDE_SET.delta({"cough_rate": 6.0})
        assert classify_anomaly(observation, baseline, WIDE_SET) is AnomalyType.RESPIRATORY

    def test_quiet_wide_observation_is_not_an_anomaly(self):
        baseline = np.array([70.0, 40.0, 15.0, 36.8, 118.0, 0.4])
        observation = baseline + WIDE_SET.delta({"systolic_bp": 1.0, "cough_rate": 0.2})
        assert classify_anomaly(observation, baseline, WIDE_SET) is None


class TestClassificationMasking:
    baseline = np.array([70.0, 40.0, 15.0, 36.8])

    def observed(self, *names: str) -> np.ndarray:
        return np.array([name in names for name in DEFAULT_CHANNEL_SET.names])

    def test_unreported_excursion_is_not_classified(self):
        observation = self.baseline + np.array([0.0, 0.0, 0.0, 1.2])
        mask = self.observed("heart_rate", "hrv_rmssd", "respiratory_rate")
        assert classify_anomaly(observation, self.baseline, DEFAULT_CHANNEL_SET, mask) is None

    def test_full_mask_matches_unmasked_classification(self):
        observation = self.baseline + np.array([15.0, -12.0, 0.0, 1.2])
        mask = np.ones(len(DEFAULT_CHANNEL_SET), dtype=np.bool_)
        assert classify_anomaly(
            observation, self.baseline, DEFAULT_CHANNEL_SET, mask
        ) is classify_anomaly(observation, self.baseline, DEFAULT_CHANNEL_SET)

    def test_missing_thermal_channel_cannot_stand_in_for_no_fever(self):
        """Respiratory distress *without fever* needs the thermal channel present."""
        observation = self.baseline + np.array([0.0, 0.0, 6.0, 0.0])
        with_thermal = self.observed("respiratory_rate", "body_temperature")
        without_thermal = self.observed("respiratory_rate")
        assert (
            classify_anomaly(observation, self.baseline, DEFAULT_CHANNEL_SET, with_thermal)
            is AnomalyType.RESPIRATORY
        )
        assert (
            classify_anomaly(observation, self.baseline, DEFAULT_CHANNEL_SET, without_thermal)
            is AnomalyType.MULTI_SYSTEM
        )

    def test_all_missing_mask_classifies_nothing(self):
        observation = self.baseline + np.array([25.0, -20.0, 8.0, 2.0])
        mask = np.zeros(len(DEFAULT_CHANNEL_SET), dtype=np.bool_)
        assert classify_anomaly(observation, self.baseline, DEFAULT_CHANNEL_SET, mask) is None


class TestExportWidth:
    def test_export_covers_channels_with_a_schema_type(self):
        timestamp = datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc)
        observation = np.array([72.0, 42.0, 15.0, 36.8, 118.0, 0.4])
        records = observation_to_records(observation, timestamp, channel_set=WIDE_SET)
        types = {record["type"] for record in records}
        assert types == {
            "heart_rate",
            "heart_rate_variability_rmssd",
            "respiratory_rate",
            "body_temperature",
            "blood_pressure_systolic",
        }
        for record in records:
            assert isinstance(record["value"], float)
            assert record["unit"]

    def test_width_mismatch_rejected(self):
        timestamp = datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="6-dimensional"):
            observation_to_records(np.zeros(4), timestamp, channel_set=WIDE_SET)
