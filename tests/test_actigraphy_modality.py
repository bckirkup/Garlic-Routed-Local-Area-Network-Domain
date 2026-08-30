"""Tests for the pedometer and sleep-motion actigraphy modality.

The two behavioural channels are the first ones whose confounders are larger
than their illness signal (an exercise bout beats sickness behaviour by an order
of magnitude on step count), so the tests here concentrate on graded response,
sign, and the negative controls that keep the modality honest rather than on
pinned values.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import CORE_VITALS, SLEEP_FRAGMENTATION_INDEX, STEP_COUNT, ChannelSet
from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.devices import (
    BASE_DEVICE_KIND,
    DEVICE_CATALOGUE,
    MOTION_ACTIGRAPHY,
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

MOTION_SET: ChannelSet = build_channel_set((BASE_DEVICE_KIND, MOTION_ACTIGRAPHY))

STEPS = STEP_COUNT.name
FRAGMENTATION = SLEEP_FRAGMENTATION_INDEX.name

# Calibration targets from docs/SENSOR_MODALITIES.md, expressed per five-minute
# epoch: a 1,500-2,500 step/day shortfall is -5.2 to -8.7 steps per epoch, and
# febrile nights add 8-16 points of fragmentation.
STEP_SHORTFALL_RANGE = (-10.0, -5.0)
FRAGMENTATION_RANGE = (8.0, 16.0)


def value(delta: np.ndarray, name: str) -> float:
    return harness.channel_value(delta, MOTION_SET, name)


def motion_model(backend: str = "rect", adoption: float = 1.0) -> GarlandModel:
    return harness.modality_model(MOTION_ACTIGRAPHY, backend=backend, adoption=adoption, seed=31)


class TestChannelWiring:
    def test_actigraphy_is_adoptable_and_widens_the_vector_by_two(self):
        assert MOTION_ACTIGRAPHY.name in DEVICE_CATALOGUE
        assert len(MOTION_SET) == len(BASE_DEVICE_KIND.channels) + 2
        assert MOTION_SET.has(STEPS)
        assert MOTION_SET.has(FRAGMENTATION)

    def test_step_count_reports_continuously_and_sleep_motion_once_a_night(self):
        steps, sleep = MOTION_ACTIGRAPHY.device_channels
        assert not steps.is_event_gated
        assert sleep.is_event_gated
        assert sleep.is_eligible(7.0)
        assert not sleep.is_eligible(15.0)
        # Motion is this device's signal, not its artifact: unlike the impedance
        # band, a moving wearer does not lose yield.
        assert steps.yield_probability(12.0, 0.3) == pytest.approx(
            steps.yield_probability(12.0, 0.0)
        )

    def test_fleet_mask_only_reports_sleep_motion_in_the_wake_epoch(self):
        fleet = DeviceFleet(
            40,
            DeviceFleetConfig(enabled=True, adoption={MOTION_ACTIGRAPHY.name: 1.0}),
            np.random.default_rng(3),
        )
        rng = np.random.default_rng(4)
        wake = fleet.observed_matrix(7.0, 0.0, rng)
        midday = fleet.observed_matrix(13.0, 0.2, rng)
        fragmentation_column = fleet.channel_set.index(FRAGMENTATION)
        step_column = fleet.channel_set.index(STEPS)
        assert wake[:, fragmentation_column].any()
        assert not midday[:, fragmentation_column].any()
        assert midday[:, step_column].any()


class TestSensitivity:
    def test_sickness_behaviour_grades_both_channels(self):
        deltas = [modality_delta(infection_axes(p), MOTION_SET) for p in (0.0, 0.25, 0.5, 1.0)]
        steps = [value(d, STEPS) for d in deltas]
        fragmentation = [value(d, FRAGMENTATION) for d in deltas]
        # Steps fall monotonically, fragmentation rises monotonically.
        assert steps == sorted(steps, reverse=True)
        assert fragmentation == sorted(fragmentation)
        assert steps[0] == pytest.approx(0.0)
        assert fragmentation[0] == pytest.approx(0.0)
        # Live knobs, not rounding: the full signature spans the calibrated range.
        low, high = STEP_SHORTFALL_RANGE
        assert low <= steps[-1] <= high
        low, high = FRAGMENTATION_RANGE
        assert low <= fragmentation[-1] <= high

    def test_prodromal_malaise_precedes_the_febrile_signature(self):
        # Behaviour is the earliest band to move: incubation already withdraws
        # activity and fragments sleep while the thermal channel is quiet.
        incubating = modality_delta(incubation_axes(1.0), MOTION_SET)
        assert value(incubating, STEPS) < 0.0
        assert value(incubating, FRAGMENTATION) > 0.0
        symptomatic = modality_delta(infection_axes(1.0), MOTION_SET)
        assert abs(value(incubating, STEPS)) < abs(value(symptomatic, STEPS))
        assert value(incubating, FRAGMENTATION) < value(symptomatic, FRAGMENTATION)

    def test_exercise_moves_steps_the_other_way_and_much_harder(self):
        exertion = modality_delta(exertion_axes(1.0), MOTION_SET)
        infection = modality_delta(infection_axes(1.0), MOTION_SET)
        assert value(exertion, STEPS) > 0.0
        assert value(infection, STEPS) < 0.0
        # A five-minute bout puts more steps in than a fully symptomatic epoch
        # takes out: the pedometer's confounder dominates its signal, which is
        # why sign matters more than magnitude for this channel.
        assert value(exertion, STEPS) > 10.0 * abs(value(infection, STEPS))

    def test_exercise_grades_with_intensity(self):
        steps = [
            value(modality_delta(exertion_axes(i), MOTION_SET), STEPS) for i in (0.0, 0.5, 1.0)
        ]
        assert steps == sorted(steps)
        assert steps[0] == pytest.approx(0.0)
        assert steps[-1] > 2.0 * steps[1] - 1e-9

    def test_a_bad_night_fragments_sleep_and_slows_the_next_day(self):
        delta = modality_delta(sleep_disruption_axes(1.0), MOTION_SET)
        assert value(delta, FRAGMENTATION) > 0.0
        assert value(delta, STEPS) < 0.0
        # Benign disruption is not a free illness detector, but it is also not
        # as heavy on the step channel as illness is.
        assert abs(value(delta, STEPS)) < abs(
            value(modality_delta(infection_axes(1.0), MOTION_SET), STEPS)
        )


class TestNegativeControls:
    def test_irritant_exposure_leaves_behaviour_alone(self):
        # An acute plume causes airway symptoms, not days of malaise: keeping
        # the behavioural channels quiet preserves toxin-versus-disease.
        delta = modality_delta(irritant_axes(1.0), MOTION_SET)
        assert value(delta, STEPS) == pytest.approx(0.0)
        assert value(delta, FRAGMENTATION) == pytest.approx(0.0)
        plume = plume_biometric_perturbation(0.8, MOTION_SET)
        assert value(plume, STEPS) == pytest.approx(0.0)
        assert value(plume, FRAGMENTATION) == pytest.approx(0.0)

    def test_contact_artifact_leaves_behaviour_alone(self):
        delta = modality_delta(contact_artifact_axes(1.0), MOTION_SET)
        assert value(delta, STEPS) == pytest.approx(0.0)
        assert value(delta, FRAGMENTATION) == pytest.approx(0.0)

    def test_core_vitals_deltas_are_unchanged_by_adopting_the_actigraph(self):
        harness.assert_core_vitals_unchanged(MOTION_SET)

    def test_core_only_fleets_see_no_behavioural_signature(self):
        delta = harness.infectious_engine().biometric_perturbation(0, 576, CORE_VITALS)
        assert delta.shape == (len(CORE_VITALS),)
        assert np.all(np.isfinite(delta))


class TestInvariants:
    def test_signatures_stay_finite_and_bounded_across_the_course(self):
        engine = harness.infectious_engine()
        for steps_since in (0, 12, 288, 576, 10_000):
            for state in (SEIRState.EXPOSED, SEIRState.INFECTIOUS, SEIRState.RECOVERED):
                engine.states[0] = state
                delta = engine.biometric_perturbation(0, steps_since, MOTION_SET)
                assert np.all(np.isfinite(delta))
                assert value(delta, STEPS) >= STEP_SHORTFALL_RANGE[0]
                assert value(delta, FRAGMENTATION) <= FRAGMENTATION_RANGE[1]

    def test_axis_ranges_are_validated(self):
        with pytest.raises(ValueError, match="activity_withdrawal"):
            IllnessAxes(activity_withdrawal=1.5)
        with pytest.raises(ValueError, match="sleep_disturbance"):
            IllnessAxes(sleep_disturbance=-2.0)

    def test_synthesised_values_stay_physical(self):
        model = harness.step_model(motion_model(), 24)
        step_column = model.channel_set.index(STEPS)
        fragmentation_column = model.channel_set.index(FRAGMENTATION)
        for agent in model.citizen_agents:
            observation = agent.last_observation
            assert np.all(np.isfinite(observation))
            assert observation[step_column] >= 0.0
            assert observation[fragmentation_column] >= 0.0


class TestModelIntegration:
    @pytest.mark.parametrize("backend", ["rect", "hex"])
    def test_model_runs_with_the_actigraph_on_both_backends(self, backend):
        harness.assert_model_runs(motion_model(backend), STEPS)

    def test_unadopted_wearers_keep_the_behavioural_channels_missing(self):
        model = motion_model(adoption=0.0)
        harness.assert_channels_structurally_missing(
            model,
            MOTION_ACTIGRAPHY,
            (STEPS, FRAGMENTATION),
            activity=0.1,
            seed=1,
        )
        assert model.device_fleet is not None
        assert not model.device_fleet.ownership[:, 1:].any()


class TestConfounderIntegration:
    def test_exercise_and_sleep_disruption_reach_the_behavioural_channels(self):
        engine = ConfounderEngine(
            16,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=1.0,
                sleep_disruption_rate=1.0,
                # Arm and fire in the same epoch: the default 8-hour delay puts
                # the disruption in the 06:00 window, which is not what is under
                # test here.
                sleep_disruption_delay_steps=1,
                sleep_disruption_delay_jitter_steps=0,
            ),
            np.random.default_rng(29),
            channel_set=MOTION_SET,
        )
        by_cause: dict[PerturbationCause, np.ndarray] = {}
        for step_index in range(4):
            step = engine.step(
                current_step=22 * 12 + step_index,
                hour_of_day=22.0,
                wearable_mask=np.ones(16, dtype=bool),
                transition_indices=set(),
            )
            for contributions in step.contributions.values():
                for contribution in contributions:
                    by_cause.setdefault(contribution.cause, contribution.delta)
        exercise = by_cause[PerturbationCause.EXERCISE]
        assert value(exercise, STEPS) > 0.0
        disrupted = by_cause[PerturbationCause.SLEEP_DISRUPTION]
        assert value(disrupted, FRAGMENTATION) > 0.0
