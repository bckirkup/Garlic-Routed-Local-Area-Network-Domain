"""Tests for cause-labelled, measurement-only biometric provenance."""

from __future__ import annotations

import numpy as np
import pytest

from garland.agents import CitizenAgent
from garland.biometric_profiles import BiometricProfile
from garland.hazards import SEIRConfig
from garland.metrics import DetectionEvent, MetricsCollector
from garland.perturbations import PerturbationCause, PerturbationContribution
from garland.privacy import AnomalyType, BroadcastQuery, EncryptedToken, PerturbedResponse
from garland.simulation import GarlandModel, SimulationConfig


def _profile() -> BiometricProfile:
    return BiometricProfile(
        resting_hr=72.0,
        resting_hrv=42.0,
        resting_rr=15.0,
        resting_temp=36.8,
    )


def test_labelled_nonhazard_perturbation_stays_out_of_hazard_booleans(monkeypatch):
    config = SimulationConfig(
        n_agents=20,
        wearable_fraction=1.0,
        n_steps=1,
        seed=42,
        plumes=[],
        seir=SEIRConfig(initial_infected=0),
    )
    model = GarlandModel(config)
    contribution = PerturbationContribution(
        PerturbationCause.EXERCISE,
        np.array([80.0, -50.0, 30.0, 0.0]),
    )
    monkeypatch.setattr(
        model,
        "_agent_perturbation_contributions",
        lambda _gidx, _concentrations: (contribution,),
    )

    tokens, _, _, _, _, _, _ = model._collect_step_tokens(
        hour_of_day=12.0,
        hour_int=12,
        month=1,
        day_of_year=15,
        time_bin=0,
        concentrations=np.zeros(20),
        activity=0.0,
    )

    assert tokens
    provenance = next(iter(model._token_provenance_lookup.values()))
    assert provenance.causes == frozenset({PerturbationCause.EXERCISE})
    assert not provenance.disease_affected
    assert not provenance.toxin_affected


def test_more_labelled_causes_increase_cause_attributed_counts():
    def count(causes: frozenset[PerturbationCause]) -> dict[str, int]:
        metrics = MetricsCollector()
        metrics.record_detection(
            DetectionEvent(
                step=0,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=True,
                agents_affected=1,
                causes=causes,
            )
        )
        return metrics.summary()["cause_attributed_detections"]["disease"]

    one = count(frozenset({PerturbationCause.EXERCISE}))
    two = count(
        frozenset(
            {PerturbationCause.EXERCISE, PerturbationCause.SLEEP_DISRUPTION}
        )
    )
    assert one["exercise"] == 1
    assert one["sleep_disruption"] == 0
    assert two["exercise"] == 1
    assert two["sleep_disruption"] == 1
    summary = MetricsCollector()
    summary.record_detection(
        DetectionEvent(
            step=0,
            hazard_type="disease",
            anomaly_type=AnomalyType.FEBRILE,
            zone_id=0,
            true_positive=True,
            agents_affected=1,
            causes=frozenset(
                {PerturbationCause.EXERCISE, PerturbationCause.SLEEP_DISRUPTION}
            ),
        )
    )
    report = summary.summary()
    assert "cause_attributed_broadcasts" not in report
    assert "disease" not in report["cause_attributed_detections"]["disease"]
    assert "toxin" not in report["cause_attributed_detections"]["toxin"]
    assert report["cause_attribution_rates"]["disease"]["exercise"] == 1.0
    assert report["cause_attribution_rates"]["disease"]["sleep_disruption"] == 1.0

    empty_rates = MetricsCollector().summary()["cause_attribution_rates"]
    assert all(
        rate is None
        for rates in empty_rates.values()
        for rate in rates.values()
    )


def test_legacy_and_labelled_perturbations_cannot_be_combined():
    agent = CitizenAgent(idx=0, has_wearable=True, profile=_profile())
    with pytest.raises(ValueError, match="cannot both be provided"):
        agent.observe_and_detect(
            hour=12,
            month=1,
            day_of_year=15,
            hour_of_day=12.0,
            rng=np.random.default_rng(123),
            cell_id=0,
            hazard_perturbation=np.zeros(4),
            perturbations=(
                PerturbationContribution(
                    PerturbationCause.EXERCISE, np.zeros(4)
                ),
            ),
        )


def test_cause_provenance_is_absent_from_protocol_objects():
    assert "causes" not in EncryptedToken._fields
    assert "causes" not in BroadcastQuery.__dataclass_fields__
    assert "causes" not in PerturbedResponse.__dataclass_fields__


def test_disease_and_toxin_sum_matches_legacy_single_vector():
    profile = _profile()
    disease = np.array([15.0, -15.0, 5.0, 1.5])
    toxin = np.array([10.0, -12.0, 12.0, 0.0])
    expected_rng = np.random.default_rng(123)
    ref_rng = np.random.default_rng(123)

    expected = model_observation(
        profile,
        expected_rng,
        hazard_perturbation=disease + toxin,
    )
    actual = model_observation(
        profile,
        ref_rng,
        perturbations=(
            PerturbationContribution(PerturbationCause.DISEASE, disease),
            PerturbationContribution(PerturbationCause.TOXIN, toxin),
        ),
    )
    np.testing.assert_array_equal(actual, expected)


def model_observation(
    profile: BiometricProfile,
    rng: np.random.Generator,
    *,
    hazard_perturbation: np.ndarray | None = None,
    perturbations: tuple[PerturbationContribution, ...] | None = None,
) -> np.ndarray:
    agent = CitizenAgent(idx=0, has_wearable=True, profile=profile)
    token = agent.observe_and_detect(
        hour=12,
        month=1,
        day_of_year=15,
        hour_of_day=12.0,
        rng=rng,
        cell_id=0,
        hazard_perturbation=hazard_perturbation,
        perturbations=perturbations,
    )
    assert token is not None
    return agent.last_observation
