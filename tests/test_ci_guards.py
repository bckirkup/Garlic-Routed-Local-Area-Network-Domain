"""CI guards for metric evidence, hazard reachability, and live mechanisms."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from garland.config import load_config_file
from garland.hazards import SEIREngine, SEIRState, plume_biometric_perturbation
from garland.privacy import AnomalyType, classify_anomaly
from garland.simulation import GarlandModel

ROOT = Path(__file__).parents[1]

_NO_EVIDENCE_KEYS = (
    "time_to_detection_disease_steps",
    "time_to_detection_disease_hours",
    "time_to_detection_toxin_steps",
    "time_to_detection_toxin_hours",
    "attributed_disease_latency_steps",
    "attributed_toxin_latency_steps",
    "coincidental_fraction_disease",
    "coincidental_fraction_toxin",
    "discrimination_score",
)


def _run_null(**privacy_overrides: int | float | str) -> GarlandModel:
    config = load_config_file(ROOT / "examples/null_baseline.yaml")
    config.n_agents = 100
    config.n_steps = 576
    for name, value in privacy_overrides.items():
        setattr(config.privacy, name, value)
    model = GarlandModel(config)
    model.run()
    return model


def _run_staged(
    *,
    n_agents: int = 300,
    n_steps: int = 1728,
    threshold_m: int = 5,
) -> GarlandModel:
    config = load_config_file(ROOT / "examples/staged_onset.yaml")
    config.n_agents = n_agents
    config.n_steps = n_steps
    config.privacy.threshold_m = threshold_m
    # This hazard-path guard preserves the historical RR operating point;
    # aggregate-mode floor behavior is exercised by the single-hazard guard.
    config.privacy.response_mechanism = "randomized_response"
    # Respondent-basis dilation is exercised separately.
    config.privacy.dilation_basis = "residents"
    model = GarlandModel(config)
    model.run()
    return model


def test_null_summary_keeps_all_no_evidence_metrics_undefined():
    summary = _run_null().metrics.summary()

    assert all(summary[key] is None for key in _NO_EVIDENCE_KEYS)


@pytest.mark.parametrize("present_hazard", ["disease", "toxin"])
def test_single_hazard_summary_does_not_invent_other_hazard_evidence(present_hazard: str):
    config = load_config_file(ROOT / "examples/staged_onset.yaml")
    config.n_agents = 100
    config.n_steps = 1728 if present_hazard == "disease" else 1200
    config.privacy.threshold_m = 2
    # These guards use the calibrated operating point; respondent-basis
    # dilation is exercised separately.
    config.privacy.dilation_basis = "residents"
    if present_hazard == "disease":
        config.plumes = []
    else:
        config.seir.outbreaks = []
        config.seir.initial_infected = 0

    model = GarlandModel(config)
    model.run()
    summary = model.metrics.summary()

    absent = "toxin" if present_hazard == "disease" else "disease"
    if present_hazard == "toxin":
        # The measured toxin cluster is below the aggregate floor: it must
        # remain visible as suppressed evidence rather than become a detection.
        assert summary["time_to_detection_toxin_steps"] is None
        assert summary["aggregate_count_below_floor_releases"] > 0
    else:
        assert summary[f"time_to_detection_{present_hazard}_steps"] is not None
    assert summary[f"time_to_detection_{absent}_steps"] is None
    assert summary[f"coincidental_fraction_{absent}"] is None
    assert summary["discrimination_score"] is None


def test_hazard_perturbations_reach_their_classification_branches():
    assert (
        classify_anomaly(plume_biometric_perturbation(0.5), np.zeros(4)) == AnomalyType.RESPIRATORY
    )

    seir = SEIREngine()
    seir.states = np.array([SEIRState.INFECTIOUS])
    early_infection = seir.biometric_perturbation(0, steps_since_infection=350)
    late_infection = seir.biometric_perturbation(0, steps_since_infection=576)
    assert classify_anomaly(early_infection, np.zeros(4)) == AnomalyType.FEBRILE
    assert classify_anomaly(late_infection, np.zeros(4)) == AnomalyType.MULTI_SYSTEM


def test_staged_run_reaches_both_hazard_detection_paths():
    model = _run_staged(threshold_m=2)
    emitted_types = {event.anomaly_type for event in model.metrics.detection_events}
    true_positive_hazards = {
        event.hazard_type for event in model.metrics.detection_events if event.true_positive
    }
    assert set(AnomalyType) <= emitted_types
    assert {"disease", "toxin"} <= true_positive_hazards


def test_threshold_and_k_anonymity_parameters_grade_operational_outputs():
    threshold_models = [
        _run_null(threshold_m=value, dilation_basis="residents") for value in (2, 5, 10)
    ]
    threshold_broadcasts = [
        model.metrics.summary()["total_broadcasts"] for model in threshold_models
    ]
    assert threshold_broadcasts == sorted(threshold_broadcasts, reverse=True)
    assert threshold_broadcasts[0] - threshold_broadcasts[-1] > 20

    # This epsilon ordering is specific to the historical RR response channel.
    k_models = [
        _run_null(
            k_min=value,
            dilation_basis="residents",
            response_mechanism="randomized_response",
        )
        for value in (1, 10, 50)
    ]
    k_epsilon = [model.metrics.summary()["total_epsilon"] for model in k_models]
    assert k_epsilon[0] < k_epsilon[1] < k_epsilon[2]


def test_randomized_response_probability_grades_privacy_outcomes():
    # This grading intentionally measures the selectable RR mechanism, not
    # the default aggregate count, whose per-release epsilon is independent
    # of randomized_response_p.
    models = [
        _run_null(
            randomized_response_p=value,
            response_mechanism="randomized_response",
        )
        for value in (0.1, 0.5, 0.9)
    ]
    responses = [model.metrics.summary()["total_responses"] for model in models]

    assert responses[0] > responses[1] > responses[2]
    assert responses[0] - responses[2] > 20


def test_laplace_scale_changes_coordinates_not_detection_outcomes():
    low = _run_staged(n_agents=100, n_steps=576, threshold_m=2)
    high_config = deepcopy(low.config)
    high_config.privacy.laplace_scale = 2000.0
    high = GarlandModel(high_config)
    high.run()

    # Issue #75: classification currently ignores reported coordinates. Keep
    # this explicit until deciding whether geo-privacy noise should affect it.
    low_events = [
        (event.step, event.hazard_type, event.anomaly_type, event.true_positive)
        for event in low.metrics.detection_events
    ]
    high_events = [
        (event.step, event.hazard_type, event.anomaly_type, event.true_positive)
        for event in high.metrics.detection_events
    ]
    assert low_events == high_events
    assert low.metrics.summary()["total_epsilon"] == high.metrics.summary()["total_epsilon"]
    low_coordinates = [
        (response.reported_x, response.reported_y) for response in low.aggregator.state.responses
    ]
    high_coordinates = [
        (response.reported_x, response.reported_y) for response in high.aggregator.state.responses
    ]
    assert low_coordinates != high_coordinates
