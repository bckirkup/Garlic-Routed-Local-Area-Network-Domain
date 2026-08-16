"""Tests for the chest electrode patch (two-lead ECG plus transit-time BP).

Three properties are specific to this modality. First, the patch re-reports two
channels the wrist already covers, so it buys *redundancy* rather than width:
the tests check that an owner whose watch has died still reports rate and
variability. Second, `ptt_systolic_bp` reads the same `arterial_stiffening` axis
as `pwv_m_s`, so the two must agree in sign rather than count as two
independent findings. Third, `ectopy_burden` is the clearest case in the fleet
of a channel a sensor fault moves about as hard as an illness does, so the tests
assert that ordering explicitly instead of pretending it separates.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.biometric_profiles import build_profile
from garland.biometric_synthesis import generate_observation_custom
from garland.channels import (
    CORE_VITALS,
    ECTOPY_BURDEN,
    HEART_RATE,
    HRV_RMSSD,
    PTT_SYSTOLIC_BP,
    PULSE_WAVE_VELOCITY,
    QTC_MS,
    ChannelSet,
    ChannelSystem,
)
from garland.devices import (
    BASE_DEVICE_KIND,
    CHEST_ELECTRODE_PATCH,
    DEVICE_CATALOGUE,
    THORACIC_EIT_ACOUSTIC_BAND,
    DeviceFleet,
    DeviceFleetConfig,
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

PATCH_SET: ChannelSet = build_channel_set((BASE_DEVICE_KIND, CHEST_ELECTRODE_PATCH))
# A fleet carrying both vascular views, for the shared-axis tests.
VASCULAR_SET: ChannelSet = build_channel_set(
    (BASE_DEVICE_KIND, CHEST_ELECTRODE_PATCH, THORACIC_EIT_ACOUSTIC_BAND)
)

QTC = QTC_MS.name
ECTOPY = ECTOPY_BURDEN.name
SYSTOLIC = PTT_SYSTOLIC_BP.name
PWV = PULSE_WAVE_VELOCITY.name

# Calibration targets from docs/SENSOR_MODALITIES.md.
QTC_RISE_RANGE = (15.0, 30.0)
SYSTOLIC_RISE_RANGE = (8.0, 18.0)


def value(delta: np.ndarray, name: str, channel_set: ChannelSet = PATCH_SET) -> float:
    return harness.channel_value(delta, channel_set, name)


def patch_model(
    backend: str = "rect",
    adoption: float = 1.0,
    lifecycle: bool = False,
) -> GarlandModel:
    return harness.modality_model(
        CHEST_ELECTRODE_PATCH,
        backend=backend,
        adoption=adoption,
        seed=53,
        lifecycle=lifecycle,
    )


class TestChannelWiring:
    def test_patch_adds_three_channels_and_re_reports_two(self):
        assert CHEST_ELECTRODE_PATCH.name in DEVICE_CATALOGUE
        assert len(PATCH_SET) == len(CORE_VITALS) + 3
        for name in (QTC, ECTOPY, SYSTOLIC):
            assert PATCH_SET.has(name)
        reported = {channel.name for channel in CHEST_ELECTRODE_PATCH.channels}
        assert {HEART_RATE.name, HRV_RMSSD.name} <= reported

    def test_channels_classify_as_cardiac_and_vascular(self):
        cardiac = PATCH_SET.system_indices(ChannelSystem.CARDIAC)
        vascular = PATCH_SET.system_indices(ChannelSystem.VASCULAR)
        assert PATCH_SET.index(QTC) in cardiac
        assert PATCH_SET.index(ECTOPY) in cardiac
        assert PATCH_SET.index(SYSTOLIC) in vascular

    def test_transit_time_is_the_most_motion_fragile_channel(self):
        yields = {
            binding.channel.name: binding.yield_probability(12.0, 0.9)
            for binding in CHEST_ELECTRODE_PATCH.device_channels
        }
        assert yields[SYSTOLIC] == min(yields.values())
        assert yields[HEART_RATE.name] == max(yields.values())
        # An R-peak survives motion that destroys a T-wave.
        assert yields[HEART_RATE.name] - yields[QTC] > 0.3

    def test_repolarisation_and_transit_time_prefer_a_sleeping_wearer(self):
        bindings = {
            binding.channel.name: binding for binding in CHEST_ELECTRODE_PATCH.device_channels
        }
        for name in (QTC, SYSTOLIC):
            binding = bindings[name]
            assert binding.yield_probability(3.0, 0.0) > binding.yield_probability(12.0, 0.0)

    def test_electrodes_keep_the_cardiac_core_alive_when_the_watch_dies(self):
        fleet = DeviceFleet(
            60,
            DeviceFleetConfig(enabled=True, adoption={CHEST_ELECTRODE_PATCH.name: 1.0}),
            np.random.default_rng(11),
        )
        active = np.ones_like(fleet.ownership)
        active[:, 0] = False  # every watch is flat; every patch still running
        observed = fleet.observed_matrix(
            12.0, 0.1, np.random.default_rng(12), subsystem_active=active.astype(np.bool_)
        )
        assert observed[:, fleet.channel_set.index(HEART_RATE.name)].mean() > 0.8
        assert observed[:, fleet.channel_set.index(HRV_RMSSD.name)].mean() > 0.7
        # Channels only the watch reports go dark, so this is redundancy on the
        # cardiac pair rather than a second whole wearable.
        assert not observed[:, fleet.channel_set.index("body_temperature")].any()


class TestSensitivity:
    def test_all_three_channels_grade_with_symptom_progress(self):
        deltas = [modality_delta(infection_axes(p), PATCH_SET) for p in (0.0, 0.25, 0.5, 1.0)]
        for name in (QTC, ECTOPY, SYSTOLIC):
            series = [value(delta, name) for delta in deltas]
            assert series == sorted(series)
            assert series[0] == pytest.approx(0.0)
            assert series[-1] > 0.0
        low, high = QTC_RISE_RANGE
        assert low <= value(deltas[-1], QTC) <= high
        low, high = SYSTOLIC_RISE_RANGE
        assert low <= value(deltas[-1], SYSTOLIC) <= high

    def test_enteric_illness_still_prolongs_the_qt_interval(self):
        """The electrolyte path is why a chest patch sees a gut infection."""
        respiratory = modality_delta(infection_axes(1.0, enteric_involvement=0.0), PATCH_SET)
        enteric = modality_delta(infection_axes(1.0, enteric_involvement=1.0), PATCH_SET)
        assert value(enteric, QTC) > value(respiratory, QTC)
        # Nothing else on the patch cares which organ the pathogen prefers.
        assert value(enteric, ECTOPY) == pytest.approx(value(respiratory, ECTOPY))
        assert value(enteric, SYSTOLIC) == pytest.approx(value(respiratory, SYSTOLIC))

    def test_transit_time_bp_tracks_the_same_axis_as_pulse_wave_velocity(self):
        for stiffening in (-0.8, -0.3, 0.0, 0.4, 1.0):
            delta = modality_delta(IllnessAxes(arterial_stiffening=stiffening), VASCULAR_SET)
            systolic = value(delta, SYSTOLIC, VASCULAR_SET)
            pwv = value(delta, PWV, VASCULAR_SET)
            assert np.sign(systolic) == np.sign(pwv)
        # Distributive shock drops both rather than reading as two findings.
        shock = modality_delta(IllnessAxes(arterial_stiffening=-0.8), VASCULAR_SET)
        assert value(shock, SYSTOLIC, VASCULAR_SET) < 0.0
        assert value(shock, PWV, VASCULAR_SET) < 0.0

    def test_an_irritant_raises_pressure_without_touching_repolarisation(self):
        irritant = modality_delta(irritant_axes(1.0), PATCH_SET)
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        assert value(irritant, SYSTOLIC) > 0.0
        # No fever means no QT prolongation and no adrenergic ectopy, so the
        # patch keeps the toxin-versus-disease separator the core vitals use.
        assert value(irritant, QTC) == pytest.approx(0.0)
        assert value(irritant, ECTOPY) == pytest.approx(0.0)
        assert value(infection, QTC) > 0.0

    def test_prodromal_signal_is_faint_on_every_channel(self):
        incubating = modality_delta(incubation_axes(1.0), PATCH_SET)
        symptomatic = modality_delta(infection_axes(1.0), PATCH_SET)
        for name in (QTC, ECTOPY, SYSTOLIC):
            fraction = value(incubating, name) / value(symptomatic, name)
            assert 0.0 < fraction < 0.35

    def test_exercise_moves_the_patch_the_same_way_illness_does(self):
        """No channel here is a free illness detector; only the size differs."""
        exertion = modality_delta(exertion_axes(1.0), PATCH_SET)
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        for name in (QTC, ECTOPY, SYSTOLIC):
            assert value(exertion, name) > 0.0
            assert value(exertion, name) < value(infection, name)

    def test_a_bad_night_leaves_the_patch_alone(self):
        delta = modality_delta(sleep_disruption_axes(1.0), PATCH_SET)
        for name in (QTC, ECTOPY, SYSTOLIC):
            assert value(delta, name) == pytest.approx(0.0)


class TestNegativeControls:
    def test_lead_noise_fakes_ectopy_and_nothing_else(self):
        delta = modality_delta(contact_artifact_axes(1.0), PATCH_SET)
        assert value(delta, ECTOPY) > 0.0
        # A loose electrode cannot invent a QT interval or a blood pressure.
        assert value(delta, QTC) == pytest.approx(0.0)
        assert value(delta, SYSTOLIC) == pytest.approx(0.0)

    def test_ectopy_alone_cannot_separate_a_fault_from_an_illness(self):
        artifact = modality_delta(contact_artifact_axes(1.0), PATCH_SET)
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        # Deliberately comparable: premature-beat burden is only informative in
        # company, when the QT and pressure channels move with it.
        assert value(artifact, ECTOPY) > 0.5 * value(infection, ECTOPY)
        assert value(infection, QTC) > 0.0
        assert value(artifact, QTC) == pytest.approx(0.0)

    def test_core_vitals_deltas_are_unchanged_by_adopting_the_patch(self):
        harness.assert_core_vitals_unchanged(PATCH_SET)


class TestInvariants:
    def test_signatures_stay_finite_and_bounded_across_the_course(self):
        engine = harness.infectious_engine()
        for steps_since in (0, 12, 288, 576, 10_000):
            for state in (SEIRState.EXPOSED, SEIRState.INFECTIOUS, SEIRState.RECOVERED):
                engine.states[0] = state
                delta = engine.biometric_perturbation(0, steps_since, PATCH_SET)
                assert np.all(np.isfinite(delta))
                assert value(delta, QTC) <= QTC_RISE_RANGE[1] + 15.0
                assert value(delta, SYSTOLIC) <= SYSTOLIC_RISE_RANGE[1]

    def test_the_qt_interval_runs_longer_at_night_than_at_midday(self):
        profile = build_profile(channel_set=PATCH_SET)
        position = PATCH_SET.index(QTC)
        rng = np.random.default_rng(29)
        night = np.mean(
            [generate_observation_custom(profile, 2.0, 40, rng)[position] for _ in range(200)]
        )
        midday = np.mean(
            [generate_observation_custom(profile, 14.0, 40, rng)[position] for _ in range(200)]
        )
        assert night > midday

    def test_synthesised_patch_values_stay_physical(self):
        model = harness.step_model(patch_model(), 24)
        columns = {name: model.channel_set.index(name) for name in (QTC, ECTOPY, SYSTOLIC)}
        for agent in model.citizen_agents:
            observation = agent.last_observation
            assert np.all(np.isfinite(observation))
            assert observation[columns[QTC]] > 250.0
            assert observation[columns[ECTOPY]] >= 0.0
            assert observation[columns[SYSTOLIC]] > 50.0


class TestModelIntegration:
    @pytest.mark.parametrize("backend", ["rect", "hex"])
    def test_model_runs_with_the_patch_on_both_backends(self, backend):
        harness.assert_model_runs(patch_model(backend), QTC)

    def test_unadopted_wearers_keep_the_patch_channels_missing(self):
        harness.assert_channels_structurally_missing(
            patch_model(adoption=0.0),
            CHEST_ELECTRODE_PATCH,
            (QTC, ECTOPY, SYSTOLIC),
            activity=0.2,
        )

    def test_patch_battery_runs_down_independently_of_the_watch(self):
        harness.assert_subsystem_battery_is_independent(
            patch_model(lifecycle=True), CHEST_ELECTRODE_PATCH
        )


class TestConfounderIntegration:
    def test_exercise_and_artifact_reach_the_patch_through_the_confounder_engine(self):
        by_cause = harness.confounder_deltas_by_cause(PATCH_SET)
        assert value(by_cause[PerturbationCause.EXERCISE], SYSTOLIC) > 0.0
        assert value(by_cause[PerturbationCause.SENSOR_ARTIFACT], ECTOPY) > 0.0
