"""Tests for the dry-electrode headband EEG (sleep architecture plus vigilance).

Two properties are specific to this modality. First, four of its five channels
are one overnight scoring pass rather than an epoch sample, so they report once
a day at wake and are the first event-gated *group* in the fleet. Second, it is
the only device whose channels disagree with each other by design: a broken
night and an infection both fragment sleep and both cost REM, and only
``slow_wave_activity_fraction`` moves in opposite directions between them, so
the tests assert that sign disagreement rather than a solo detector.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import (
    ALPHA_THETA_RATIO,
    CORE_VITALS,
    REM_SLEEP_FRACTION,
    SLEEP_FRAGMENTATION_INDEX,
    SLEEP_ONSET_LATENCY,
    SLOW_WAVE_ACTIVITY_FRACTION,
    WAKE_AFTER_SLEEP_ONSET,
    ChannelSet,
    ChannelSystem,
)
from garland.devices import (
    BASE_DEVICE_KIND,
    DEVICE_CATALOGUE,
    HEADBAND_EEG,
    MOTION_ACTIGRAPHY,
    build_channel_set,
)
from garland.hazards import SEIRState
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

BAND_SET: ChannelSet = build_channel_set((BASE_DEVICE_KIND, HEADBAND_EEG))
# A wearer carrying both sleep views, for the shared-axis tests.
SLEEP_SET: ChannelSet = build_channel_set((BASE_DEVICE_KIND, HEADBAND_EEG, MOTION_ACTIGRAPHY))

ONSET = SLEEP_ONSET_LATENCY.name
WASO = WAKE_AFTER_SLEEP_ONSET.name
REM = REM_SLEEP_FRACTION.name
SLOW_WAVE = SLOW_WAVE_ACTIVITY_FRACTION.name
ALPHA_THETA = ALPHA_THETA_RATIO.name
FRAGMENTATION = SLEEP_FRAGMENTATION_INDEX.name

BAND_CHANNELS = (ONSET, WASO, REM, SLOW_WAVE, ALPHA_THETA)
# The four channels scored from one night's recording, as opposed to sampled.
OVERNIGHT_CHANNELS = (ONSET, WASO, REM, SLOW_WAVE)

# Calibration targets from docs/SENSOR_MODALITIES.md.
ONSET_RISE_RANGE = (15.0, 25.0)
WASO_RISE_RANGE = (20.0, 30.0)
REM_LOSS_RANGE = (-8.0, -5.0)
SLOW_WAVE_RISE_RANGE = (4.0, 9.0)


def value(delta: np.ndarray, name: str, channel_set: ChannelSet = BAND_SET) -> float:
    return harness.channel_value(delta, channel_set, name)


def band_model(
    backend: str = "rect",
    adoption: float = 1.0,
    lifecycle: bool = False,
) -> GarlandModel:
    return harness.modality_model(
        HEADBAND_EEG,
        backend=backend,
        adoption=adoption,
        seed=67,
        lifecycle=lifecycle,
    )


class TestChannelWiring:
    def test_headband_adds_five_channels(self):
        assert HEADBAND_EEG.name in DEVICE_CATALOGUE
        assert len(BAND_SET) == len(CORE_VITALS) + 5
        for name in BAND_CHANNELS:
            assert BAND_SET.has(name)

    def test_channels_classify_as_sleep_and_neural(self):
        sleep = BAND_SET.system_indices(ChannelSystem.SLEEP)
        neural = BAND_SET.system_indices(ChannelSystem.NEURAL)
        for name in OVERNIGHT_CHANNELS:
            assert BAND_SET.index(name) in sleep
        # The one waking channel is the only spectral measure, so it is the only
        # one that reads as neural rather than as sleep architecture.
        assert BAND_SET.index(ALPHA_THETA) in neural

    def test_the_overnight_channels_report_only_once_a_day(self):
        bindings = {binding.channel.name: binding for binding in HEADBAND_EEG.device_channels}
        for name in OVERNIGHT_CHANNELS:
            binding = bindings[name]
            assert binding.is_event_gated
            assert binding.is_eligible(7.0)
            for hour in (1.0, 12.0, 18.0, 23.0):
                assert not binding.is_eligible(hour)
        # A staging pass over one recording, so the four complete together
        # rather than drifting apart across the day.
        completions = {bindings[name].event_completion_hours for name in OVERNIGHT_CHANNELS}
        assert len(completions) == 1

    def test_the_vigilance_ratio_is_awake_only_and_motion_fragile(self):
        binding = next(
            item for item in HEADBAND_EEG.device_channels if item.channel.name == ALPHA_THETA
        )
        assert not binding.is_event_gated
        still = binding.yield_probability(12.0, 0.0)
        walking = binding.yield_probability(12.0, 0.9)
        assert still > walking
        # A dry forehead electrode under a frown or a stride returns nothing
        # usable, and overnight the wearer is not awake to have a ratio at all.
        assert walking == pytest.approx(0.0)
        assert binding.yield_probability(3.0, 0.0) < still

    def test_the_headband_charges_and_is_removed_faster_than_the_watch(self):
        assert HEADBAND_EEG.power.removal_multiplier > 1.0
        assert HEADBAND_EEG.power.charge_multiplier > 1.0
        assert HEADBAND_EEG.power.drain_multiplier > 1.0


class TestSensitivity:
    def test_every_channel_grades_with_symptom_progress(self):
        deltas = [modality_delta(infection_axes(p), BAND_SET) for p in (0.0, 0.25, 0.5, 1.0)]
        for name in (ONSET, WASO, SLOW_WAVE):
            series = [value(delta, name) for delta in deltas]
            assert series == sorted(series)
            assert series[0] == pytest.approx(0.0)
            assert series[-1] > 0.0
        for name in (REM, ALPHA_THETA):
            series = [value(delta, name) for delta in deltas]
            assert series == sorted(series, reverse=True)
            assert series[0] == pytest.approx(0.0)
            assert series[-1] < 0.0

    def test_illness_deviations_land_in_the_calibrated_ranges(self):
        delta = modality_delta(infection_axes(1.0), BAND_SET)
        for name, (low, high) in (
            (ONSET, ONSET_RISE_RANGE),
            (WASO, WASO_RISE_RANGE),
            (REM, REM_LOSS_RANGE),
            (SLOW_WAVE, SLOW_WAVE_RISE_RANGE),
        ):
            assert low <= value(delta, name) <= high

    def test_slow_wave_power_is_the_only_channel_that_changes_sign(self):
        """The headband's actual contribution to the joint score.

        An infection intensifies slow-wave sleep and a ruined night suppresses
        it, while both fragment the night and both cost REM. No single channel
        separates them; the disagreement between them does.
        """
        infection = modality_delta(infection_axes(1.0), BAND_SET)
        bad_night = modality_delta(sleep_disruption_axes(1.0), BAND_SET)
        assert value(infection, SLOW_WAVE) > 0.0
        assert value(bad_night, SLOW_WAVE) < 0.0
        for name in (ONSET, WASO, REM, ALPHA_THETA):
            assert np.sign(value(infection, name)) == np.sign(value(bad_night, name))

    def test_slow_wave_loss_from_a_bad_night_beats_the_gain_from_illness(self):
        """Asymmetric on purpose: a wrecked night costs more than fever adds."""
        gains = [
            value(modality_delta(IllnessAxes(slow_wave_drive=d), BAND_SET), SLOW_WAVE)
            for d in (0.25, 0.5, 1.0)
        ]
        losses = [
            value(modality_delta(IllnessAxes(slow_wave_drive=-d), BAND_SET), SLOW_WAVE)
            for d in (0.25, 0.5, 1.0)
        ]
        assert gains == sorted(gains)
        assert losses == sorted(losses, reverse=True)
        assert abs(losses[-1]) > gains[-1]

    def test_rem_loss_grades_without_separating_cause(self):
        series = [
            value(modality_delta(IllnessAxes(rem_suppression=level), BAND_SET), REM)
            for level in (0.0, 0.3, 0.6, 1.0)
        ]
        assert series == sorted(series, reverse=True)
        assert series[0] == pytest.approx(0.0)
        # Both an infection and a broken night reach a large fraction of the
        # full-scale loss, so REM alone is not an illness finding.
        infection = value(modality_delta(infection_axes(1.0), BAND_SET), REM)
        bad_night = value(modality_delta(sleep_disruption_axes(1.0), BAND_SET), REM)
        assert bad_night < 0.5 * infection

    def test_the_headband_and_the_watch_agree_about_a_fragmented_night(self):
        """One sleep_disturbance state, two devices, no double counting."""
        for level in (-0.4, 0.0, 0.3, 1.0):
            delta = modality_delta(IllnessAxes(sleep_disturbance=level), SLEEP_SET)
            waso = value(delta, WASO, SLEEP_SET)
            fragmentation = value(delta, FRAGMENTATION, SLEEP_SET)
            assert np.sign(waso) == np.sign(fragmentation)

    def test_prodromal_sleep_change_is_faint_but_present(self):
        incubating = modality_delta(incubation_axes(1.0), BAND_SET)
        symptomatic = modality_delta(infection_axes(1.0), BAND_SET)
        for name in BAND_CHANNELS:
            fraction = value(incubating, name) / value(symptomatic, name)
            assert 0.0 < fraction < 0.6

    def test_an_irritant_plume_leaves_the_night_alone(self):
        irritant = modality_delta(irritant_axes(1.0), BAND_SET)
        # An acute exposure has no night to disturb, so the four staging
        # channels stay flat and remain a toxin-versus-disease separator.
        for name in OVERNIGHT_CHANNELS:
            assert value(irritant, name) == pytest.approx(0.0)
        # Waking inattention is real, though, so the spectral ratio still moves.
        assert value(irritant, ALPHA_THETA) < 0.0

    def test_exercise_deepens_slow_wave_sleep_the_same_way_illness_does(self):
        exertion = modality_delta(exertion_axes(1.0), BAND_SET)
        infection = modality_delta(infection_axes(1.0), BAND_SET)
        assert 0.0 < value(exertion, SLOW_WAVE) < value(infection, SLOW_WAVE)
        # A hard bout settles the night rather than fragmenting it, which is the
        # one place the benign and illness arms disagree in sign.
        assert value(exertion, WASO) < 0.0
        assert value(infection, WASO) > 0.0


class TestNegativeControls:
    def test_a_lifting_electrode_fakes_wake_and_flattens_the_spectrum(self):
        delta = modality_delta(contact_artifact_axes(1.0), BAND_SET)
        # An unstageable epoch scores as wake, so a poor contact manufactures
        # wake after sleep onset out of nothing.
        assert value(delta, WASO) > 0.0
        assert value(delta, ALPHA_THETA) < 0.0
        # It cannot invent a sleep latency or a stage distribution, though.
        assert value(delta, ONSET) == pytest.approx(0.0)
        assert value(delta, REM) == pytest.approx(0.0)
        assert value(delta, SLOW_WAVE) == pytest.approx(0.0)

    def test_the_spectral_ratio_alone_cannot_separate_a_fault_from_an_illness(self):
        artifact = modality_delta(contact_artifact_axes(1.0), BAND_SET)
        infection = modality_delta(infection_axes(1.0), BAND_SET)
        # Deliberately comparable: alpha/theta is only informative in company.
        assert abs(value(artifact, ALPHA_THETA)) > 0.4 * abs(value(infection, ALPHA_THETA))
        assert value(infection, SLOW_WAVE) > 0.0
        assert value(artifact, SLOW_WAVE) == pytest.approx(0.0)

    def test_core_vitals_deltas_are_unchanged_by_adopting_the_headband(self):
        harness.assert_core_vitals_unchanged(BAND_SET)


class TestInvariants:
    def test_signatures_stay_finite_and_bounded_across_the_course(self):
        engine = harness.infectious_engine()
        for steps_since in (0, 12, 288, 576, 10_000):
            for state in (SEIRState.EXPOSED, SEIRState.INFECTIOUS, SEIRState.RECOVERED):
                engine.states[0] = state
                delta = engine.biometric_perturbation(0, steps_since, BAND_SET)
                assert np.all(np.isfinite(delta))
                assert value(delta, ONSET) <= ONSET_RISE_RANGE[1]
                assert value(delta, WASO) <= WASO_RISE_RANGE[1]
                assert value(delta, REM) >= REM_LOSS_RANGE[0]
                assert value(delta, SLOW_WAVE) <= SLOW_WAVE_RISE_RANGE[1]

    def test_synthesised_headband_values_stay_physical(self):
        model = harness.step_model(band_model(), 24)
        columns = {name: model.channel_set.index(name) for name in BAND_CHANNELS}
        for agent in model.citizen_agents:
            observation = agent.last_observation
            assert np.all(np.isfinite(observation))
            assert observation[columns[ONSET]] >= 0.0
            assert observation[columns[WASO]] >= 0.0
            # Stage shares are percentages of a night, and a spectral ratio of
            # band powers cannot be negative.
            for name in (REM, SLOW_WAVE):
                assert 0.0 <= observation[columns[name]] <= 100.0
            assert observation[columns[ALPHA_THETA]] > 0.0

    def test_axis_validation_accepts_the_signed_slow_wave_arm(self):
        assert IllnessAxes(slow_wave_drive=-1.0).slow_wave_drive == pytest.approx(-1.0)
        with pytest.raises(ValueError):
            IllnessAxes(rem_suppression=-0.5)


class TestModelIntegration:
    @pytest.mark.parametrize("backend", ["rect", "hex"])
    def test_model_runs_with_the_headband_on_both_backends(self, backend):
        harness.assert_model_runs(band_model(backend), REM)

    def test_unadopted_wearers_keep_the_headband_channels_missing(self):
        harness.assert_channels_structurally_missing(
            band_model(adoption=0.0),
            HEADBAND_EEG,
            BAND_CHANNELS,
            hour_of_day=7.0,
            activity=0.0,
        )

    def test_headband_battery_runs_down_independently_of_the_watch(self):
        harness.assert_subsystem_battery_is_independent(band_model(lifecycle=True), HEADBAND_EEG)


class TestConfounderIntegration:
    def test_a_bad_night_and_an_artifact_reach_the_headband(self):
        by_cause = harness.confounder_deltas_by_cause(BAND_SET)
        artifact = by_cause[PerturbationCause.SENSOR_ARTIFACT]
        assert value(artifact, WASO) > 0.0
        assert value(artifact, ALPHA_THETA) < 0.0
