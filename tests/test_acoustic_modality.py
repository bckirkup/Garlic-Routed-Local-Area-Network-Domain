"""Tests for the respiratory acoustic patch (cough, speech, breath, heart sounds).

Three properties are specific to this modality and drive most of these tests.
First, cough is the one channel an irritant plume moves *harder* than an
infection does, so it is deliberately not a toxin-versus-disease discriminator
and the tests assert that ordering rather than the usual "toxin leaves it
alone". Second, wheeze and crackle are the discriminating pair: an irritant
narrows the conducting airway and leaves the alveoli silent, while a pneumonia
cracks, so neither channel alone says what happened. Third,
`crackle_count_per_cycle` is driven by real consolidation *and* by garment
shear, so the artifact arm has to be checked against the channels it must
*not* reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import (
    CORE_VITALS,
    COUGH_RATE,
    CRACKLE_COUNT_PER_CYCLE,
    HEART_SOUND_S1_S2_RATIO,
    S3_ENERGY_FRACTION,
    SPEECH_PAUSE_RATIO,
    WHEEZE_DURATION_FRACTION,
    ChannelSet,
    ChannelSystem,
)
from garland.devices import (
    BASE_DEVICE_KIND,
    DEVICE_CATALOGUE,
    RESPIRATORY_ACOUSTIC_PATCH,
    DeviceFleet,
    DeviceFleetConfig,
    build_channel_set,
)
from garland.hazards import SEIRState, plume_biometric_perturbation
from garland.modality_signatures import (
    IllnessAxes,
    cardiac_decompensation_axes,
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

PATCH_SET: ChannelSet = build_channel_set((BASE_DEVICE_KIND, RESPIRATORY_ACOUSTIC_PATCH))

COUGH = COUGH_RATE.name
SPEECH = SPEECH_PAUSE_RATIO.name
WHEEZE = WHEEZE_DURATION_FRACTION.name
CRACKLES = CRACKLE_COUNT_PER_CYCLE.name
HEART_SOUND = HEART_SOUND_S1_S2_RATIO.name
S3 = S3_ENERGY_FRACTION.name

PATCH_CHANNELS = (COUGH, SPEECH, WHEEZE, CRACKLES, HEART_SOUND, S3)
# Channels a respiratory infection is expected to move at all. S3 needs raised
# filling pressures, which an infection does not produce.
INFECTION_CHANNELS = (COUGH, SPEECH, WHEEZE, CRACKLES, HEART_SOUND)

# Calibration targets from docs/SENSOR_MODALITIES.md.
COUGH_RISE_RANGE = (10.0, 20.0)
CRACKLE_RISE_RANGE = (4.0, 12.0)
WHEEZE_RISE_RANGE = (0.15, 0.45)
S3_RISE_RANGE = (5.0, 12.0)
S1_S2_FALL_RANGE = (-0.65, -0.45)


def value(delta: np.ndarray, name: str) -> float:
    return harness.channel_value(delta, PATCH_SET, name)


def patch_model(
    backend: str = "rect",
    adoption: float = 1.0,
    lifecycle: bool = False,
) -> GarlandModel:
    return harness.modality_model(
        RESPIRATORY_ACOUSTIC_PATCH,
        backend=backend,
        adoption=adoption,
        seed=41,
        lifecycle=lifecycle,
    )


class TestChannelWiring:
    def test_patch_is_adoptable_and_widens_the_vector_by_its_bundle(self):
        assert RESPIRATORY_ACOUSTIC_PATCH.name in DEVICE_CATALOGUE
        assert len(PATCH_SET) == len(CORE_VITALS) + len(PATCH_CHANNELS)
        for name in PATCH_CHANNELS:
            assert PATCH_SET.has(name)

    def test_channels_classify_as_respiratory_and_cardiac_not_as_acoustics(self):
        respiratory = PATCH_SET.system_indices(ChannelSystem.RESPIRATORY)
        cardiac = PATCH_SET.system_indices(ChannelSystem.CARDIAC)
        for name in (COUGH, SPEECH, WHEEZE, CRACKLES):
            assert PATCH_SET.index(name) in respiratory
        for name in (HEART_SOUND, S3):
            assert PATCH_SET.index(name) in cardiac

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
        awake = [speech.yield_probability(12.0, activity) for activity in (0.0, 0.4, 1.0)]
        assert awake == sorted(awake)
        assert speech.yield_probability(3.0, 0.0) == pytest.approx(0.0)
        for name in (WHEEZE, CRACKLES):
            binding = bindings[name]
            assert binding.yield_probability(3.0, 0.0) > binding.yield_probability(12.0, 0.0)

    def test_a_tonal_wheeze_survives_more_masking_than_a_transient_crackle(self):
        bindings = {
            binding.channel.name: binding for binding in RESPIRATORY_ACOUSTIC_PATCH.device_channels
        }
        for activity in (0.0, 0.5, 1.0):
            wheeze = bindings[WHEEZE].yield_probability(12.0, activity)
            crackle = bindings[CRACKLES].yield_probability(12.0, activity)
            assert wheeze > crackle

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
    def test_every_infection_channel_grades_with_symptom_progress(self):
        deltas = [modality_delta(infection_axes(p), PATCH_SET) for p in (0.0, 0.25, 0.5, 1.0)]
        for name in INFECTION_CHANNELS:
            series = [value(delta, name) for delta in deltas]
            assert series == sorted(series)
            assert series[0] == pytest.approx(0.0)
            assert series[-1] > 0.0
        low, high = COUGH_RISE_RANGE
        assert low <= value(deltas[-1], COUGH) <= high
        low, high = CRACKLE_RISE_RANGE
        assert low <= value(deltas[-1], CRACKLES) <= high

    def test_a_pneumonia_cracks_far_harder_than_it_wheezes(self):
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        # Both in units of their own excursion cut, which is the only way to
        # compare a count per breath with a fraction of a cycle.
        crackle_cuts = value(infection, CRACKLES) / CRACKLE_COUNT_PER_CYCLE.deviation_threshold
        wheeze_cuts = value(infection, WHEEZE) / WHEEZE_DURATION_FRACTION.deviation_threshold
        assert crackle_cuts > 1.5 * wheeze_cuts

    def test_bronchospasm_reaches_the_calibrated_wheeze_range(self):
        wheeze = value(modality_delta(IllnessAxes(airway_obstruction=1.0), PATCH_SET), WHEEZE)
        low, high = WHEEZE_RISE_RANGE
        assert low <= wheeze <= high

    def test_decompensation_drops_s1_s2_and_raises_s3_without_a_fever(self):
        decompensated = modality_delta(cardiac_decompensation_axes(1.0), PATCH_SET)
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        low, high = S1_S2_FALL_RANGE
        assert low <= value(decompensated, HEART_SOUND) <= high
        low, high = S3_RISE_RANGE
        assert low <= value(decompensated, S3) <= high
        # Febrile inotropy moves the same ratio the other way, so the sign is
        # the discriminator and neither direction alone means "ill".
        assert value(infection, HEART_SOUND) > 0.0
        assert value(infection, S3) == pytest.approx(0.0)
        # Oedema cracks too, so crackles cannot say *why* the alveoli filled.
        assert value(decompensated, CRACKLES) > 0.0

    def test_dysfunction_moves_the_heart_sound_further_than_inotropy_does(self):
        dysfunction = modality_delta(IllnessAxes(cardiac_contractility=-1.0), PATCH_SET)
        inotropy = modality_delta(IllnessAxes(cardiac_contractility=1.0), PATCH_SET)
        assert abs(value(dysfunction, HEART_SOUND)) > abs(value(inotropy, HEART_SOUND))
        assert value(dysfunction, HEART_SOUND) < 0.0 < value(inotropy, HEART_SOUND)

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
        assert value(enteric, CRACKLES) < value(respiratory, CRACKLES)
        assert value(enteric, WHEEZE) < value(respiratory, WHEEZE)
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

    def test_an_irritant_wheezes_with_silent_alveoli(self):
        irritant = modality_delta(irritant_axes(1.0), PATCH_SET)
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        # Bronchospasm without consolidation: the wheeze exceeds a pneumonia's
        # while the crackle count stays at exactly zero, which is what excludes
        # parenchymal involvement.
        assert value(irritant, WHEEZE) > value(infection, WHEEZE)
        assert value(irritant, CRACKLES) == pytest.approx(0.0)

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
        for name in PATCH_CHANNELS:
            assert value(delta, name) == pytest.approx(0.0)


class TestNegativeControls:
    def test_friction_artifact_fakes_crackles_only(self):
        delta = modality_delta(contact_artifact_axes(1.0), PATCH_SET)
        assert value(delta, CRACKLES) > 0.0
        # Garment shear makes short transients, not tonal ones, so it cannot
        # fake a wheeze; nor can it invent a cough, a pause pattern in speech,
        # or a heart-sound amplitude.
        for name in (WHEEZE, COUGH, HEART_SOUND, S3):
            assert value(delta, name) == pytest.approx(0.0)
        # Real consolidation moves the crackle count considerably harder, which
        # is what keeps it usable despite the shared driver.
        infection = modality_delta(infection_axes(1.0), PATCH_SET)
        assert value(infection, CRACKLES) > 2.0 * value(delta, CRACKLES)

    def test_artifact_does_not_fake_speech_fragmentation(self):
        delta = modality_delta(contact_artifact_axes(1.0), PATCH_SET)
        assert value(delta, SPEECH) == pytest.approx(0.0)

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
                assert value(delta, COUGH) <= COUGH_RISE_RANGE[1]
                assert value(delta, CRACKLES) <= CRACKLE_RISE_RANGE[1]
                assert value(delta, WHEEZE) <= WHEEZE_RISE_RANGE[1]

    def test_the_new_axis_ranges_are_validated(self):
        for name in ("airway_irritation", "airway_obstruction", "parenchymal_consolidation"):
            with pytest.raises(ValueError, match=name):
                IllnessAxes(**{name: -0.1})
            with pytest.raises(ValueError, match=name):
                IllnessAxes(**{name: 2.0})
        # Contractility is signed, so only the magnitude is bounded.
        assert IllnessAxes(cardiac_contractility=-1.0).cardiac_contractility == pytest.approx(-1.0)
        with pytest.raises(ValueError, match="cardiac_contractility"):
            IllnessAxes(cardiac_contractility=-1.5)

    def test_synthesised_patch_values_stay_physical(self):
        model = harness.step_model(patch_model(), 24)
        columns = {
            name: model.channel_set.index(name) for name in (COUGH, SPEECH, WHEEZE, CRACKLES, S3)
        }
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
        harness.assert_model_runs(patch_model(backend), COUGH)

    def test_unadopted_wearers_keep_the_patch_channels_missing(self):
        harness.assert_channels_structurally_missing(
            patch_model(adoption=0.0),
            RESPIRATORY_ACOUSTIC_PATCH,
            PATCH_CHANNELS,
            activity=0.2,
        )

    def test_patch_battery_runs_down_independently_of_the_watch(self):
        harness.assert_subsystem_battery_is_independent(
            patch_model(lifecycle=True), RESPIRATORY_ACOUSTIC_PATCH
        )


class TestConfounderIntegration:
    def test_exercise_and_artifact_reach_the_patch_through_the_confounder_engine(self):
        by_cause = harness.confounder_deltas_by_cause(PATCH_SET)
        assert value(by_cause[PerturbationCause.EXERCISE], SPEECH) > 0.0
        assert value(by_cause[PerturbationCause.SENSOR_ARTIFACT], CRACKLES) > 0.0
