"""CI guards for metric evidence, hazard reachability, and live mechanisms."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from garland.config import load_config_file
from garland.hazards import (
    SEIREngine,
    SEIRState,
    compute_plume_concentrations,
    plume_biometric_perturbation,
)
from garland.privacy import AnomalyType, classify_anomaly
from garland.simulation import GarlandModel

ROOT = Path(__file__).parents[1]
GUARD_N_AGENTS = 2_000
GUARD_WEARABLE_FRACTION = 0.6
GUARD_GRID_M = 1_800.0
GUARD_PLUME_START = 96
GUARD_PLUME_DURATION = 288
GUARD_OUTBREAK_START = 288
COMMITTED_N_AGENTS = 10_000
COMMITTED_WEARABLE_FRACTION = 0.15
COMMITTED_GRID_M = 2_000.0

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
    n_steps: int = 576,
    threshold_m: int = 5,
) -> GarlandModel:
    config = load_config_file(ROOT / "examples/staged_onset.yaml")
    _configure_staged_guard(config, n_steps=n_steps, threshold_m=threshold_m)
    model = GarlandModel(config)
    model.run()
    return model


def _configure_staged_guard(
    config,
    *,
    n_steps: int,
    threshold_m: int,
) -> None:
    """Use a dense, plume-fitting downscale for staged hazard guards."""
    config.n_agents = GUARD_N_AGENTS
    config.n_steps = n_steps
    config.wearable_fraction = GUARD_WEARABLE_FRACTION
    config.grid_width = GUARD_GRID_M
    config.grid_height = GUARD_GRID_M
    config.privacy.threshold_m = threshold_m
    config.privacy.dilation_basis = "residents"
    config.plumes[0].source_x = GUARD_GRID_M / 2.0
    config.plumes[0].source_y = GUARD_GRID_M / 2.0
    config.plumes[0].start_step = GUARD_PLUME_START
    config.plumes[0].duration_steps = GUARD_PLUME_DURATION
    for outbreak in config.seir.outbreaks:
        outbreak.center_x = GUARD_GRID_M / 2.0
        outbreak.center_y = GUARD_GRID_M / 2.0
        outbreak.start_step = GUARD_OUTBREAK_START


def test_null_summary_keeps_all_no_evidence_metrics_undefined():
    summary = _run_null().metrics.summary()

    assert all(summary[key] is None for key in _NO_EVIDENCE_KEYS)


@pytest.mark.parametrize("present_hazard", ["disease", "toxin"])
def test_single_hazard_summary_does_not_invent_other_hazard_evidence(present_hazard: str):
    config = load_config_file(ROOT / "examples/staged_onset.yaml")
    _configure_staged_guard(config, n_steps=576, threshold_m=2)
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
        assert summary["time_to_detection_toxin_steps"] is not None
        assert summary["aggregate_count_evidence_releases"] > 0
    else:
        assert summary[f"time_to_detection_{present_hazard}_steps"] is not None
    assert summary[f"time_to_detection_{absent}_steps"] is None
    assert summary[f"coincidental_fraction_{absent}"] is None
    assert summary["discrimination_score"] is None


def test_aggregate_toxin_cluster_clears_floor_and_detects():
    config = load_config_file(ROOT / "examples/staged_onset.yaml")
    _configure_staged_guard(config, n_steps=576, threshold_m=2)
    config.seir.outbreaks = []
    config.seir.initial_infected = 0

    model = GarlandModel(config)
    model.run()
    summary = model.metrics.summary()
    releases = summary["aggregate_count_releases"]
    evidence_threshold = summary["aggregate_count_evidence_threshold"]

    assert summary["time_to_detection_toxin_steps"] is not None
    floor_clearing_releases = [
        row for row in releases if row["true_count_evaluation_only"] >= evidence_threshold + 1
    ]
    assert floor_clearing_releases
    assert any(row["released_count"] > evidence_threshold for row in floor_clearing_releases)
    toxin_events = [
        event for event in model.metrics.detection_events if event.hazard_type == "toxin"
    ]
    assert toxin_events
    assert any(event.true_positive for event in toxin_events)


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
    low = _run_staged(n_steps=576, threshold_m=2)
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


def test_staged_guard_density_stays_near_committed_scenario() -> None:
    config = load_config_file(ROOT / "examples/staged_onset.yaml")
    _configure_staged_guard(config, n_steps=576, threshold_m=2)
    guard_density = (
        config.n_agents
        * config.wearable_fraction
        / (config.grid_width * config.grid_height / 1_000_000.0)
    )
    committed_density = (
        COMMITTED_N_AGENTS
        * COMMITTED_WEARABLE_FRACTION
        / (COMMITTED_GRID_M * COMMITTED_GRID_M / 1_000_000.0)
    )

    assert guard_density >= 300.0
    assert committed_density == pytest.approx(375.0)
    assert 0.8 <= guard_density / committed_density <= 1.2


def test_staged_guard_plume_fits_inside_grid() -> None:
    config = load_config_file(ROOT / "examples/staged_onset.yaml")
    _configure_staged_guard(config, n_steps=576, threshold_m=2)
    axis = np.arange(0.0, config.grid_width + 1_000.0, 10.0)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    concentrations, _ = compute_plume_concentrations(
        grid_x.ravel(),
        grid_y.ravel(),
        config.plumes,
        GUARD_PLUME_START,
    )
    above_gate = concentrations.reshape(grid_x.shape) > config.toxin_exposure_concentration_gate()
    assert np.any(above_gate)
    assert np.all(grid_x[above_gate] <= config.grid_width)
    assert np.all(grid_y[above_gate] <= config.grid_height)
    downwind_extent = np.max(grid_x[above_gate]) - config.plumes[0].source_x
    assert downwind_extent > 0.0
    assert (config.grid_width / 2.0) / downwind_extent >= 1.2
