"""Tests for the respiratory acoustic patch (cough, speech, breath, heart sounds).

Two properties are specific to this modality and drive most of these tests.
First, cough is the one channel an irritant plume moves *harder* than an
infection does, so it is deliberately not a toxin-versus-disease discriminator
and the tests assert that ordering rather than the usual "toxin leaves it
alone". Second, `adventitious_breath_fraction` is driven by both real
consolidation and instrument artifact, so the artifact arm has to be checked
against the channels it must *not* reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import (
    ADVENTITIOUS_BREATH_FRACTION,
    CORE_VITALS,
    COUGH_RATE,
    HEART_SOUND_S1_S2_RATIO,
    SPEECH_PAUSE_RATIO,
    ChannelSet,
    ChannelSystem,
)
from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.device_lifecycle import DeviceLifecycleConfig
from garland.devices import (
    BASE_DEVICE_KIND,
    DEVICE_CATALOGUE,
    RESPIRATORY_ACOUSTIC_PATCH,
    DeviceFleet,
    DeviceFleetConfig,
    build_channel_set,
)
from garland.hazards import SEIRConfig, SEIREngine, SEIRState, plume_biometric_perturbation
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
from garland.simulation import GarlandModel, SimulationConfig

PATCH_SET: ChannelSet = build_channel_set((BASE_DEVICE_KIND, RESPIRATORY_ACOUSTIC_PATCH))

COUGH = COUGH_RATE.name
SPEECH = SPEECH_PAUSE_RATIO.name
BREATH = ADVENTITIOUS_BREATH_FRACTION.name
HEART_SOUND = HEART_SOUND_S1_S2_RATIO.name

# Calibration targets from docs/SENSOR_MODALITIES.md.
COUGH_RISE_RANGE = (10.0, 20.0)
BREATH_RISE_RANGE = (12.0, 25.0)


def value(delta: np.ndarray, name: str) -> float:
    return float(delta[PATCH_SET.index(name)])


def infectious_engine() -> SEIREngine:
    engine = SEIREngine(SEIRConfig(initial_infected=1))
    engine.initialize(8, np.random.default_rng(3))
    engine.states[0] = SEIRState.INFECTIOUS
    return engine


def patch_model(
    backend: str = "rect",
    adoption: float = 1.0,
    seed: int = 41,
    lifecycle: bool = False,
) -> GarlandModel:
    return GarlandModel(
        SimulationConfig(
            n_agents=200,
            n_steps=20,
            wearable_fraction=0.5,
            seed=seed,
            spatial_backend=backend,
            device_lifecycle=DeviceLifecycleConfig(enabled=lifecycle),
            devices=DeviceFleetConfig(
                enabled=True,
                adoption={RESPIRATORY_ACOUSTIC_PATCH.name: adoption},
            ),
        )
    )


class TestChannelWiring:
    def test_patch_is_adoptable_and_widens_the_vector_by_four(self):
        assert RESPIRATORY_ACOUSTIC_PATCH.name in DEVICE_CATALOGUE
        assert len(PATCH_SET) == len(CORE_VITALS) + 4
        for name in (COUGH, SPEECH, BREATH, HEART_SOUND):
            assert PATCH_SET.has(name)

    def test_channels_classify_as_respiratory_and_cardiac_not_as_acoustics(self):
        respiratory = PATCH_SET.system_indices(ChannelSystem.RESPIRATORY)
        cardiac = PATCH_SET.system_indices(ChannelSystem.CARDIAC)
        for name in (COUGH, SPEECH, BREATH):
            assert PATCH_SET.index(name) in respiratory
        assert PATCH_SET.index(HEART_SOUND) in cardiac

    def test_heart_sounds_are_the_most_motion_fragile_channel_on_the_patch(self):
        yields = {
            binding.channel.name: binding.yield_probability(12.0, 0.9)
            for binding in RESPIRATORY_ACOUSTIC_PATCH.device_channels
        }
        assert yields[HEART_SOUND] == min(yields.values())
        assert yields[COUGH] == max(yields.values())
        # A cough is loud enough to survive motion that buries a heart sound.
        assert yields[COUGH] - yields[HEART_SOUND] > 0.4

    def test_speech_needs_a_waking_wearer_and_breath_sounds_prefer_a_sleeping_one(self):
        bindings = {
            binding.channel.name: binding for binding in RESPIRATORY_ACOUSTIC_PATCH.device_channels
        }
        speech = bindings[SPEECH]
        breath = bindings[BREATH]
        awake = [speech.yield_probability(12.0, activity) for activity in (0.0, 0.4, 1.0)]
        assert awake == sorted(awake)
        assert speech.yield_probability(3.0, 0.0) == pytest.approx(0.0)
        assert breath.yield_probability(3.0, 0.0) > breath.yield_probability(12.0, 0.0)

    def test_fleet_reports_cough_far_more_often_than_heart_sounds_while_active(self):
        fleet = DeviceFleet(
            80,
            DeviceFleetConfig(enabled=True, adoption={RESPIRATORY_ACOUSTIC_PATCH.name: 1.0}),
            np.random.default_rng(7),
        )
        observed = fleet.observed_matrix(12.0, 0.8, np.random.default_rng(8))
        cough_rate = observed[:, fleet.channel_set.index(COUGH)].mean()
        heart_rate = observed[:, fleet.channel_set.index(HEART_SOUND)].mean()
        speech_rate = observed[:, fleet.channel_set.index(SPEECH)].mean()
        assert cough_rate > 0.6
        assert heart_rate < 0.3
        assert speech_rate > 0.2


class TestSensitivity:
    def test_all_four_channels_grade_with_symptom_progress(self):
        deltas = [modality_delta(infection_axes(p), PATCH_SET) for p in (0.0, 0.25, 0.5, 1.0)]
        for name in (COUGH, SPEECH, BREATH, HEART_SOUND):
            series = [value(delta, name) for delta in deltas]
            assert series == sorted(series)
            assert series[0] == pytest.approx(0.0)
            assert series[-1] > 0.0
        low, high = COUGH_RISE_RANGE
        assert low <= value(deltas[-1], COUGH) <= high
        low, high = BREATH_RISE_RANGE
        assert low <= value(deltas[-1], BREATH) <= high

    def test_cough_precedes_the_febrile_channels(self):
        incubating = modality_delta(incubation_axes(1.0), PATCH_SET)
        symptomatic = modality_delta(infection_axes(1.0), PATCH_SET)
        cough_fraction = value(incubating, COUGH) / value(symptomatic, COUGH)
        heart_fraction = value(incubating, HEART_SOUND) / value(symptomatic, HEART_SOUND)
        assert value(incubating, COUGH) > 0.0
        assert cough_fraction > heart_fraction

    def test_enteric_tropism_quietens_the_patch(self):
        respiratory = modality_delta(infection_axes(1.0, enteric_involvement=0.0), PATCH_SET)
        mixed = modality_delta(infection_axes(1.0, enteric_involvement=0.5), PATCH_SET)
        enteric = modality_delta(infection_axes(1.0, enteric_involvement=0.9), PATCH_SET)
        coughs = [value(delta, COUGH) for delta in (enteric, mixed, respiratory)]
        assert coughs == sorted(coughs)
        assert value(enteric, BREATH) < value(respiratory, BREATH)
        # Systemic inflammation is unchanged by tropism, so the heart-sound
        # channel does not care which organ the virus prefers.
        assert value(enteric, HEART_SOUND) == pytest.approx(value(respiratory, HEART_SOUND))

    def test_an_irritant_plume_coughs_harder_than_a_fever_without_the_fever(self):
        irritant = modality_delta(irritant_axes(1.0), PATCH_SET)
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        assert value(irritant, COUGH) > value(infection, COUGH)
        # The separator is not the cough: it is the absent inflammatory drive.
        assert value(irritant, HEART_SOUND) == pytest.approx(0.0)
        assert value(infection, HEART_SOUND) > 0.0

    def test_plume_exposure_reaches_the_patch_through_the_hazard_path(self):
        deltas = [plume_biometric_perturbation(dose, PATCH_SET) for dose in (0.0, 0.3, 0.6, 1.0)]
        coughs = [value(delta, COUGH) for delta in deltas]
        assert coughs == sorted(coughs)
        assert coughs[0] == pytest.approx(0.0)
        assert coughs[-1] > 1.0

    def test_exercise_fragments_speech_without_provoking_a_cough(self):
        exertion = modality_delta(exertion_axes(1.0), PATCH_SET)
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        assert value(exertion, SPEECH) > 0.0
        assert value(exertion, SPEECH) < value(infection, SPEECH)
        assert value(exertion, COUGH) < 0.2 * value(infection, COUGH)

    def test_a_bad_night_leaves_the_patch_alone(self):
        delta = modality_delta(sleep_disruption_axes(1.0), PATCH_SET)
        for name in (COUGH, SPEECH, BREATH, HEART_SOUND):
            assert value(delta, name) == pytest.approx(0.0)


class TestNegativeControls:
    def test_friction_artifact_fakes_crackles_only(self):
        delta = modality_delta(contact_artifact_axes(1.0), PATCH_SET)
        assert value(delta, BREATH) > 0.0
        # A contact microphone rubbing on a shirt cannot invent a cough, a
        # pause pattern in speech, or a heart-sound amplitude ratio.
        assert value(delta, COUGH) == pytest.approx(0.0)
        assert value(delta, HEART_SOUND) == pytest.approx(0.0)
        # Real consolidation moves the same channel considerably harder, which
        # is what keeps it usable despite the shared driver.
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        assert value(infection, BREATH) > 2.0 * value(delta, BREATH)

    def test_artifact_does_not_fake_speech_fragmentation(self):
        delta = modality_delta(contact_artifact_axes(1.0), PATCH_SET)
        assert value(delta, SPEECH) == pytest.approx(0.0)

    def test_core_vitals_deltas_are_unchanged_by_adopting_the_patch(self):
        core = infectious_engine().biometric_perturbation(0, 400, CORE_VITALS)
        wide = infectious_engine().biometric_perturbation(0, 400, PATCH_SET)
        for position, name in enumerate(CORE_VITALS.names):
            assert wide[PATCH_SET.index(name)] == pytest.approx(core[position])


class TestInvariants:
    def test_signatures_stay_finite_and_bounded_across_the_course(self):
        engine = infectious_engine()
        for steps_since in (0, 12, 288, 576, 10_000):
            for state in (SEIRState.EXPOSED, SEIRState.INFECTIOUS, SEIRState.RECOVERED):
                engine.states[0] = state
                delta = engine.biometric_perturbation(0, steps_since, PATCH_SET)
                assert np.all(np.isfinite(delta))
                assert value(delta, COUGH) <= COUGH_RISE_RANGE[1]
                assert value(delta, BREATH) <= BREATH_RISE_RANGE[1]

    def test_the_new_axis_range_is_validated(self):
        with pytest.raises(ValueError, match="airway_irritation"):
            IllnessAxes(airway_irritation=-0.1)
        with pytest.raises(ValueError, match="airway_irritation"):
            IllnessAxes(airway_irritation=2.0)

    def test_synthesised_patch_values_stay_physical(self):
        model = patch_model()
        for _ in range(24):
            model.step()
        columns = {name: model.channel_set.index(name) for name in (COUGH, SPEECH, BREATH)}
        heart_column = model.channel_set.index(HEART_SOUND)
        for agent in model.citizen_agents:
            observation = agent.last_observation
            assert np.all(np.isfinite(observation))
            for column in columns.values():
                assert observation[column] >= 0.0
            assert observation[columns[SPEECH]] <= 100.0
            assert observation[heart_column] > 0.0


class TestModelIntegration:
    @pytest.mark.parametrize("backend", ["rect", "hex"])
    def test_model_runs_with_the_patch_on_both_backends(self, backend):
        model = patch_model(backend)
        assert model.device_fleet is not None
        assert model.channel_set.has(COUGH)
        for _ in range(18):
            model.step()
        assert model.citizen_agents
        for agent in model.citizen_agents:
            assert np.all(np.isfinite(agent.baseline.ema))

    def test_unadopted_wearers_keep_the_patch_channels_missing(self):
        model = patch_model(adoption=0.0)
        assert model.device_fleet is not None
        for _ in range(6):
            model.step()
        assert model.device_fleet.owner_counts()[RESPIRATORY_ACOUSTIC_PATCH.name] == 0
        mask = model.device_fleet.observed_matrix(12.0, 0.2, np.random.default_rng(4))
        for name in (COUGH, SPEECH, BREATH, HEART_SOUND):
            assert not mask[:, model.channel_set.index(name)].any()

    def test_patch_battery_runs_down_independently_of_the_watch(self):
        model = patch_model(lifecycle=True)
        lifecycle = model.subsystem_lifecycle
        assert lifecycle is not None
        engine = lifecycle.engines[RESPIRATORY_ACOUSTIC_PATCH.name]
        for _ in range(12):
            model.step()
        assert np.all(np.isfinite(engine.battery_levels))
        assert np.all(engine.battery_levels >= 0.0)
        watch_batteries = np.array(
            [agent.battery_level for agent in model.citizen_agents if agent.has_wearable]
        )
        assert not np.allclose(engine.battery_levels[: watch_batteries.size], watch_batteries)


class TestConfounderIntegration:
    def test_exercise_and_artifact_reach_the_patch_through_the_confounder_engine(self):
        engine = ConfounderEngine(
            16,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=1.0,
                sensor_artifact_probability=1.0,
            ),
            np.random.default_rng(19),
            channel_set=PATCH_SET,
        )
        by_cause: dict[PerturbationCause, np.ndarray] = {}
        for step_index in range(4):
            step = engine.step(
                current_step=12 * 12 + step_index,
                hour_of_day=12.0,
                wearable_mask=np.ones(16, dtype=bool),
                transition_indices=set(range(16)),
            )
            for contributions in step.contributions.values():
                for contribution in contributions:
                    by_cause.setdefault(contribution.cause, contribution.delta)
        assert value(by_cause[PerturbationCause.EXERCISE], SPEECH) > 0.0
        assert value(by_cause[PerturbationCause.SENSOR_ARTIFACT], BREATH) > 0.0
