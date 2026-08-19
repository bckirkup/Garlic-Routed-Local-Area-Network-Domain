"""Tests for the instrumented-footwear gait modality.

Gait is the first ambulation-*gated* modality: a shoe reports nothing for a
sedentary epoch and a lot during a walking bout, which is the inverse of the
impedance band's artifact-limited yield. It is also the first modality carrying
a channel (`gait_asymmetry`) that no infection touches, so these tests lean on
sign disagreement and negative controls rather than pinned magnitudes.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import (
    CORE_VITALS,
    GAIT_ASYMMETRY,
    GAIT_SPEED,
    STRIDE_TIME_VARIABILITY,
    ChannelSet,
)
from garland.devices import (
    BASE_DEVICE_KIND,
    DEVICE_CATALOGUE,
    INSTRUMENTED_FOOTWEAR,
    DeviceFleet,
    DeviceFleetConfig,
    build_channel_set,
)
from garland.hazards import SEIRState, plume_biometric_perturbation
from garland.modality_signatures import (
    IllnessAxes,
    contact_artifact_axes,
    exertion_axes,
    incubation_axes,
    infection_axes,
    irritant_axes,
    modality_delta,
    sleep_disruption_axes,
)
from garland.perturbations import PerturbationCause
from garland.simulation import GarlandModel
from tests import modality_harness as harness

GAIT_SET: ChannelSet = build_channel_set((BASE_DEVICE_KIND, INSTRUMENTED_FOOTWEAR))

SPEED = GAIT_SPEED.name
STRIDE_CV = STRIDE_TIME_VARIABILITY.name
ASYMMETRY = GAIT_ASYMMETRY.name

# Calibration targets from docs/SENSOR_MODALITIES.md: malaise costs roughly one
# minimal-clinically-important difference of habitual speed, and fatigue about
# doubles a 2.4% stride-time CV.
SPEED_LOSS_RANGE = (-0.25, -0.10)
STRIDE_CV_RISE_RANGE = (1.5, 3.5)


def value(delta: np.ndarray, name: str) -> float:
    return harness.channel_value(delta, GAIT_SET, name)


def gait_model(
    backend: str = "rect",
    adoption: float = 1.0,
    lifecycle: bool = False,
) -> GarlandModel:
    return harness.modality_model(
        INSTRUMENTED_FOOTWEAR,
        backend=backend,
        adoption=adoption,
        seed=37,
        lifecycle=lifecycle,
    )


class TestChannelWiring:
    def test_footwear_is_adoptable_and_widens_the_vector_by_three(self):
        assert INSTRUMENTED_FOOTWEAR.name in DEVICE_CATALOGUE
        assert len(GAIT_SET) == len(CORE_VITALS) + 3
        for name in (SPEED, STRIDE_CV, ASYMMETRY):
            assert GAIT_SET.has(name)

    def test_yield_rises_with_ambulation_and_falls_to_nothing_overnight(self):
        speed = INSTRUMENTED_FOOTWEAR.device_channels[0]
        walking = [speed.yield_probability(12.0, activity) for activity in (0.0, 0.3, 0.6, 1.0)]
        assert walking == sorted(walking)
        # Meaningful separation, not merely ordered: a walking bout is worth more
        # than half a duty cycle over sitting still.
        assert walking[-1] - walking[0] > 0.5
        assert speed.yield_probability(3.0, 0.0) == pytest.approx(0.0)
        assert not speed.is_event_gated

    def test_asymmetry_needs_more_strides_than_speed_does(self):
        speed, stride_cv, asymmetry = INSTRUMENTED_FOOTWEAR.device_channels
        yields = [binding.yield_probability(12.0, 0.5) for binding in (speed, stride_cv, asymmetry)]
        assert yields == sorted(yields, reverse=True)

    def test_fleet_mask_reports_gait_when_walking_and_not_when_asleep(self):
        fleet = DeviceFleet(
            60,
            DeviceFleetConfig(enabled=True, adoption={INSTRUMENTED_FOOTWEAR.name: 1.0}),
            np.random.default_rng(5),
        )
        rng = np.random.default_rng(6)
        speed_column = fleet.channel_set.index(SPEED)
        walking = fleet.observed_matrix(12.0, 0.9, rng)
        asleep = fleet.observed_matrix(3.0, 0.0, rng)
        assert walking[:, speed_column].mean() > 0.5
        assert not asleep[:, speed_column].any()


class TestSensitivity:
    def test_illness_slows_gait_and_destabilises_it_monotonically(self):
        deltas = [modality_delta(infection_axes(p), GAIT_SET) for p in (0.0, 0.25, 0.5, 1.0)]
        speeds = [value(d, SPEED) for d in deltas]
        stride_cvs = [value(d, STRIDE_CV) for d in deltas]
        assert speeds == sorted(speeds, reverse=True)
        assert stride_cvs == sorted(stride_cvs)
        assert speeds[0] == pytest.approx(0.0)
        assert stride_cvs[0] == pytest.approx(0.0)
        low, high = SPEED_LOSS_RANGE
        assert low <= speeds[-1] <= high
        low, high = STRIDE_CV_RISE_RANGE
        assert low <= stride_cvs[-1] <= high

    def test_prodromal_gait_change_precedes_the_febrile_signature(self):
        incubating = modality_delta(incubation_axes(1.0), GAIT_SET)
        symptomatic = modality_delta(infection_axes(1.0), GAIT_SET)
        assert value(incubating, SPEED) < 0.0
        assert value(incubating, STRIDE_CV) > 0.0
        assert abs(value(incubating, SPEED)) < abs(value(symptomatic, SPEED))
        assert value(incubating, STRIDE_CV) < value(symptomatic, STRIDE_CV)

    def test_exercise_and_illness_disagree_on_sign_of_speed_only(self):
        exertion = modality_delta(exertion_axes(1.0), GAIT_SET)
        infection = modality_delta(infection_axes(1.0), GAIT_SET)
        # This pair is the whole point of splitting the two gait axes: a bout
        # raises speed *and* variability, illness lowers speed and raises
        # variability, so neither channel discriminates alone.
        assert value(exertion, SPEED) > 0.0
        assert value(infection, SPEED) < 0.0
        assert value(exertion, STRIDE_CV) > 0.0
        assert value(infection, STRIDE_CV) > 0.0

    def test_exercise_grades_with_intensity(self):
        speeds = [value(modality_delta(exertion_axes(i), GAIT_SET), SPEED) for i in (0.0, 0.5, 1.0)]
        assert speeds == sorted(speeds)
        assert speeds[-1] - speeds[1] > 0.1

    def test_a_bad_night_destabilises_gait_less_than_illness_does(self):
        disrupted = modality_delta(sleep_disruption_axes(1.0), GAIT_SET)
        infection = modality_delta(infection_axes(1.0), GAIT_SET)
        assert value(disrupted, STRIDE_CV) > 0.0
        assert value(disrupted, STRIDE_CV) < value(infection, STRIDE_CV)
        assert value(disrupted, SPEED) < 0.0


class TestNegativeControls:
    def test_no_illness_state_touches_asymmetry(self):
        for axes in (
            incubation_axes(1.0),
            infection_axes(1.0),
            infection_axes(1.0, enteric_involvement=1.0),
            irritant_axes(1.0),
            exertion_axes(1.0),
            sleep_disruption_axes(1.0),
        ):
            assert value(modality_delta(axes, GAIT_SET), ASYMMETRY) == pytest.approx(0.0)

    def test_only_instrument_artifact_moves_asymmetry(self):
        delta = modality_delta(contact_artifact_axes(1.0), GAIT_SET)
        assert value(delta, ASYMMETRY) > 0.0
        # And artifact is not allowed to fake the physiological gait channels.
        assert value(delta, SPEED) == pytest.approx(0.0)
        assert value(delta, STRIDE_CV) == pytest.approx(0.0)

    def test_irritant_exposure_leaves_gait_alone(self):
        delta = modality_delta(irritant_axes(1.0), GAIT_SET)
        plume = plume_biometric_perturbation(0.8, GAIT_SET)
        for name in (SPEED, STRIDE_CV, ASYMMETRY):
            assert value(delta, name) == pytest.approx(0.0)
            assert value(plume, name) == pytest.approx(0.0)

    def test_core_vitals_deltas_are_unchanged_by_adopting_footwear(self):
        harness.assert_core_vitals_unchanged(GAIT_SET)


class TestInvariants:
    def test_signatures_stay_finite_and_bounded_across_the_course(self):
        engine = harness.infectious_engine()
        for steps_since in (0, 12, 288, 576, 10_000):
            for state in (SEIRState.EXPOSED, SEIRState.INFECTIOUS, SEIRState.RECOVERED):
                engine.states[0] = state
                delta = engine.biometric_perturbation(0, steps_since, GAIT_SET)
                assert np.all(np.isfinite(delta))
                assert value(delta, SPEED) >= SPEED_LOSS_RANGE[0]
                assert value(delta, STRIDE_CV) <= STRIDE_CV_RISE_RANGE[1]

    def test_new_axis_ranges_are_validated(self):
        with pytest.raises(ValueError, match="neuromotor_fatigue"):
            IllnessAxes(neuromotor_fatigue=-0.5)
        with pytest.raises(ValueError, match="instrument_artifact"):
            IllnessAxes(instrument_artifact=1.5)

    def test_synthesised_gait_values_stay_physical(self):
        model = harness.step_model(gait_model(), 24)
        columns = [model.channel_set.index(name) for name in (SPEED, STRIDE_CV, ASYMMETRY)]
        for agent in model.citizen_agents:
            observation = agent.last_observation
            assert np.all(np.isfinite(observation))
            for column in columns:
                assert observation[column] >= 0.0
            assert observation[columns[0]] < 5.0


class TestModelIntegration:
    @pytest.mark.parametrize("backend", ["rect", "hex"])
    def test_model_runs_with_footwear_on_both_backends(self, backend):
        harness.assert_model_runs(gait_model(backend), SPEED)

    def test_unadopted_wearers_keep_the_gait_channels_missing(self):
        harness.assert_channels_structurally_missing(
            gait_model(adoption=0.0),
            INSTRUMENTED_FOOTWEAR,
            (SPEED, STRIDE_CV, ASYMMETRY),
            activity=0.9,
            seed=2,
        )

    def test_footwear_battery_is_independent_of_the_watch(self):
        # Different hardware, different cell: the shoe's small battery does not
        # track the watch's.
        harness.assert_subsystem_battery_is_independent(
            gait_model(lifecycle=True), INSTRUMENTED_FOOTWEAR
        )


class TestConfounderIntegration:
    def test_sensor_artifact_reaches_asymmetry_and_exercise_reaches_speed(self):
        by_cause = harness.confounder_deltas_by_cause(GAIT_SET, seed=23)
        assert value(by_cause[PerturbationCause.EXERCISE], SPEED) > 0.0
        assert value(by_cause[PerturbationCause.SENSOR_ARTIFACT], ASYMMETRY) > 0.0
