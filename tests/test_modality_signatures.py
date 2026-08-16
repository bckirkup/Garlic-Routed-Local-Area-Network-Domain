"""Tests for illness/confounder signatures on the EIT/acoustic band channels."""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import CORE_VITALS, ChannelSet
from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.devices import (
    ABDOMINAL_ACOUSTIC_BAND,
    BASE_DEVICE_KIND,
    THORACIC_EIT_ACOUSTIC_BAND,
    build_channel_set,
)
from garland.hazards import (
    SEIRConfig,
    SEIREngine,
    SEIRState,
    plume_biometric_perturbation,
)
from garland.modality_signatures import (
    IllnessAxes,
    contact_artifact_axes,
    exertion_axes,
    incubation_axes,
    infection_axes,
    irritant_axes,
    modality_delta,
)
from garland.pathogens import get_pathogen, seir_config_from_pathogen
from garland.perturbations import PerturbationCause
from garland.privacy import AnomalyType, classify_anomaly

BAND_SET: ChannelSet = build_channel_set(
    (BASE_DEVICE_KIND, THORACIC_EIT_ACOUSTIC_BAND, ABDOMINAL_ACOUSTIC_BAND)
)

# Calibration ranges from docs/SENSOR_MODALITIES.md, as (low, high) illness
# deviations that a fully ramped signature must land inside.
ILLNESS_RANGES: dict[str, tuple[float, float]] = {
    "regional_ventilation_heterogeneity": (0.20, 0.40),
    "pep_ms": (-35.0, -20.0),
    "pwv_m_s": (1.5, 2.8),
    "bowel_sound_burst_rate": (12.0, 25.0),
    "gastric_emptying_index": (35.0, 65.0),
}


def band_value(delta: np.ndarray, name: str) -> float:
    return float(delta[BAND_SET.index(name)])


def infectious_engine(enteric_involvement: float = 0.0) -> SEIREngine:
    engine = SEIREngine(SEIRConfig(initial_infected=1, enteric_involvement=enteric_involvement))
    engine.initialize(8, np.random.default_rng(5))
    engine.states[0] = SEIRState.INFECTIOUS
    return engine


class TestSignatureCalibration:
    def test_respiratory_infection_lands_in_the_calibrated_ranges(self):
        delta = modality_delta(infection_axes(1.0), BAND_SET)
        for name in ("regional_ventilation_heterogeneity", "pep_ms", "pwv_m_s"):
            low, high = ILLNESS_RANGES[name]
            assert low <= band_value(delta, name) <= high
        low, high = ILLNESS_RANGES["gastric_emptying_index"]
        assert low <= band_value(delta, "gastric_emptying_index") <= high

    def test_enteric_infection_lands_in_the_calibrated_bowel_range(self):
        delta = modality_delta(infection_axes(1.0, enteric_involvement=1.0), BAND_SET)
        low, high = ILLNESS_RANGES["bowel_sound_burst_rate"]
        assert low <= band_value(delta, "bowel_sound_burst_rate") <= high

    def test_enteric_tropism_trades_ventilation_for_gut_motility(self):
        respiratory = modality_delta(infection_axes(1.0), BAND_SET)
        enteric = modality_delta(infection_axes(1.0, enteric_involvement=0.9), BAND_SET)
        assert band_value(enteric, "bowel_sound_burst_rate") > band_value(
            respiratory, "bowel_sound_burst_rate"
        )
        assert band_value(enteric, "regional_ventilation_heterogeneity") < band_value(
            respiratory, "regional_ventilation_heterogeneity"
        )
        # Systemic inflammation is unchanged: the tropism moves where the
        # signature shows up, not how sick the person is.
        assert band_value(enteric, "pep_ms") == pytest.approx(band_value(respiratory, "pep_ms"))

    def test_ileus_and_shock_move_the_opposite_way(self):
        delta = modality_delta(IllnessAxes(enteric_drive=-1.0, arterial_stiffening=-1.0), BAND_SET)
        assert band_value(delta, "bowel_sound_burst_rate") == pytest.approx(-4.5)
        assert -2.5 <= band_value(delta, "pwv_m_s") <= -1.5

    def test_graded_severity_grades_every_band_channel(self):
        magnitudes = [
            np.abs(modality_delta(infection_axes(progress, 0.5), BAND_SET))
            for progress in (0.0, 0.25, 0.5, 1.0)
        ]
        for earlier, later in zip(magnitudes, magnitudes[1:]):
            assert np.all(later >= earlier)
            assert float(later.sum()) > float(earlier.sum())
        assert float(magnitudes[0].sum()) == pytest.approx(0.0)

    def test_incubation_is_a_faint_fraction_of_symptomatic(self):
        incubating = np.abs(modality_delta(incubation_axes(1.0), BAND_SET))
        symptomatic = np.abs(modality_delta(infection_axes(1.0), BAND_SET))
        assert 0.0 < float(incubating.sum()) < 0.3 * float(symptomatic.sum())

    def test_axes_outside_their_range_are_rejected(self):
        with pytest.raises(ValueError, match="inflammatory_drive"):
            IllnessAxes(inflammatory_drive=1.5)
        with pytest.raises(ValueError, match="pulmonary_involvement"):
            IllnessAxes(pulmonary_involvement=-0.1)
        with pytest.raises(ValueError, match="enteric_drive"):
            IllnessAxes(enteric_drive=-2.0)

    def test_shared_axis_means_one_cause_moves_pwv_once(self):
        # PWV and any future pulse-transit-time channel read one arterial state,
        # so a fever cannot be counted twice through two vascular channels.
        axes = infection_axes(1.0)
        assert axes.arterial_stiffening == pytest.approx(axes.inflammatory_drive)


class TestDifferentiability:
    def test_toxin_leaves_the_febrile_band_channels_alone(self):
        delta = modality_delta(irritant_axes(1.0), BAND_SET)
        assert band_value(delta, "regional_ventilation_heterogeneity") > 0.0
        assert band_value(delta, "pep_ms") == pytest.approx(0.0)
        assert band_value(delta, "gastric_emptying_index") == pytest.approx(0.0)
        assert band_value(delta, "bowel_sound_burst_rate") == pytest.approx(0.0)

    def test_exercise_mimics_part_of_a_fever_on_the_bands(self):
        exertion = modality_delta(exertion_axes(1.0), BAND_SET)
        infection = modality_delta(infection_axes(1.0), BAND_SET)
        # Not a free discriminator: exertion shortens PEP and stiffens arteries.
        assert band_value(exertion, "pep_ms") < 0.0
        assert band_value(exertion, "pwv_m_s") > 0.0
        # But the ventilation field stays far more even than in consolidation.
        vent = "regional_ventilation_heterogeneity"
        assert band_value(exertion, vent) < 0.5 * band_value(infection, vent)

    def test_contact_artifact_only_corrupts_the_impedance_field(self):
        delta = modality_delta(contact_artifact_axes(1.0), BAND_SET)
        assert band_value(delta, "regional_ventilation_heterogeneity") > 0.0
        for name in ("pep_ms", "pwv_m_s", "bowel_sound_burst_rate", "gastric_emptying_index"):
            assert band_value(delta, name) == pytest.approx(0.0)

    def test_wide_infection_and_toxin_classify_as_before(self):
        resting = np.zeros(len(BAND_SET))
        seir = infectious_engine()
        infection = seir.biometric_perturbation(0, 576, BAND_SET)
        toxin = plume_biometric_perturbation(0.5, BAND_SET)
        assert classify_anomaly(infection, resting, BAND_SET) in (
            AnomalyType.FEBRILE,
            AnomalyType.MULTI_SYSTEM,
        )
        # Ventilation heterogeneity is a respiratory channel and the irritant
        # signature still carries no fever, so the discriminator survives.
        assert classify_anomaly(toxin, resting, BAND_SET) == AnomalyType.RESPIRATORY


class TestHazardIntegration:
    def test_core_vitals_signatures_are_untouched_by_the_bands(self):
        seir_core = infectious_engine()
        seir_band = infectious_engine()
        core = seir_core.biometric_perturbation(0, 400, CORE_VITALS)
        wide = seir_band.biometric_perturbation(0, 400, BAND_SET)
        for position, name in enumerate(CORE_VITALS.names):
            assert wide[BAND_SET.index(name)] == pytest.approx(core[position])
        plume_core = plume_biometric_perturbation(0.7, CORE_VITALS)
        plume_wide = plume_biometric_perturbation(0.7, BAND_SET)
        for position, name in enumerate(CORE_VITALS.names):
            assert plume_wide[BAND_SET.index(name)] == pytest.approx(plume_core[position])

    def test_core_only_fleets_see_no_band_signature(self):
        seir = infectious_engine(enteric_involvement=1.0)
        delta = seir.biometric_perturbation(0, 576, CORE_VITALS)
        assert delta.shape == (len(CORE_VITALS),)
        assert np.all(np.isfinite(delta))

    def test_band_signatures_are_finite_and_bounded_across_the_course(self):
        seir = infectious_engine(enteric_involvement=0.5)
        for steps in (0, 12, 288, 576, 10_000):
            for state in (SEIRState.EXPOSED, SEIRState.INFECTIOUS, SEIRState.RECOVERED):
                seir.states[0] = state
                delta = seir.biometric_perturbation(0, steps, BAND_SET)
                assert np.all(np.isfinite(delta))
                assert band_value(delta, "gastric_emptying_index") <= 65.0
                assert band_value(delta, "pwv_m_s") <= 2.8
                assert band_value(delta, "pep_ms") >= -35.0

    def test_recovered_agents_have_no_signature(self):
        seir = infectious_engine()
        seir.states[0] = SEIRState.RECOVERED
        assert not np.any(seir.biometric_perturbation(0, 576, BAND_SET))

    def test_enteric_involvement_is_validated_and_carried_by_the_library(self):
        with pytest.raises(ValueError, match="enteric_involvement"):
            SEIRConfig(enteric_involvement=1.5)
        norovirus = get_pathogen("norovirus")
        assert float(norovirus.seir["enteric_involvement"]) > 0.5
        assert seir_config_from_pathogen("norovirus").enteric_involvement > 0.5
        influenza = seir_config_from_pathogen("influenza_seasonal")
        assert influenza.enteric_involvement == pytest.approx(0.0)


class TestConfounderIntegration:
    def test_exercise_and_artifacts_reach_adopted_band_channels(self):
        engine = ConfounderEngine(
            16,
            ConfoundersConfig(enabled=True, exercise_rate=1.0, sensor_artifact_probability=1.0),
            np.random.default_rng(19),
            channel_set=BAND_SET,
        )
        step = engine.step(
            current_step=0,
            hour_of_day=12.0,
            wearable_mask=np.ones(16, dtype=bool),
            transition_indices={0, 1},
        )
        by_cause: dict[PerturbationCause, np.ndarray] = {}
        for contributions in step.contributions.values():
            for contribution in contributions:
                by_cause.setdefault(contribution.cause, contribution.delta)
        exercise = by_cause[PerturbationCause.EXERCISE]
        artifact = by_cause[PerturbationCause.SENSOR_ARTIFACT]
        assert band_value(exercise, "pep_ms") < 0.0
        assert band_value(artifact, "regional_ventilation_heterogeneity") > 0.0
        assert band_value(artifact, "pep_ms") == pytest.approx(0.0)

    def test_confounder_deltas_stay_core_only_for_a_core_fleet(self):
        engine = ConfounderEngine(
            8,
            ConfoundersConfig(enabled=True, exercise_rate=1.0),
            np.random.default_rng(23),
            channel_set=CORE_VITALS,
        )
        step = engine.step(
            current_step=0,
            hour_of_day=12.0,
            wearable_mask=np.ones(8, dtype=bool),
            transition_indices=set(),
        )
        for contributions in step.contributions.values():
            for contribution in contributions:
                assert contribution.delta.shape == (len(CORE_VITALS),)
                assert np.all(np.isfinite(contribution.delta))
