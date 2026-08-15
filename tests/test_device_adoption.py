"""Tests for first-time wearable adoption schedules."""

from __future__ import annotations

import pytest

from garland.config import load_config_file
from garland.simulation import GarlandModel

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _config(mode: str, *, backend: str = "hex"):
    config = load_config_file(ROOT / "examples/null_baseline.yaml")
    config.n_agents = 40
    config.wearable_fraction = 1.0
    config.n_steps = 20
    config.world_settling_steps = 5
    config.baseline_warmup_steps = 0
    config.spatial_backend = backend
    config.mobility_model = "static"
    config.adoption.mode = mode
    config.adoption.start_step = 6
    config.adoption.initial_adopted_fraction = 0.8
    config.adoption.rate = 0.3
    return config


def test_default_adoption_matches_explicit_startup_schedule():
    implicit = load_config_file(ROOT / "examples/null_baseline.yaml")
    explicit = load_config_file(ROOT / "examples/null_baseline.yaml")
    implicit.n_agents = explicit.n_agents = 40
    implicit.n_steps = explicit.n_steps = 20
    implicit.mobility_model = explicit.mobility_model = "static"
    explicit.adoption.mode = "all_at_start"
    first = GarlandModel(implicit)
    second = GarlandModel(explicit)
    first.run()
    second.run()

    for key in ("total_broadcasts", "total_responses", "fleet_cold_start"):
        assert first.metrics.summary()[key] == second.metrics.summary()[key]
    assert first.metrics.summary()["adoption_events"] == []


@pytest.mark.parametrize("backend", ["hex", "rect"])
def test_adoption_counts_partition_wearables_each_step(backend: str):
    model = GarlandModel(_config("trickle", backend=backend))
    wearable_count = len(model.citizen_agents)
    model.run()

    rows = model.metrics.step_records
    assert all(
        row["not_adopted_wearables"] + row["adopted_wearables"] == wearable_count for row in rows
    )
    assert all(0 <= row["not_adopted_wearables"] <= wearable_count for row in rows)
    assert model.metrics.summary()["adoption_events"]


def test_trickle_rate_grades_onboarding_cold_fraction():
    fractions = []
    for rate in (0.05, 0.2, 0.5):
        config = _config("trickle")
        config.adoption.rate = rate
        model = GarlandModel(config)
        model.run()
        fraction = model.metrics.summary()[
            "post_world_settling_cold_baseline_wearable_step_fraction"
        ]
        assert fraction is not None
        fractions.append(fraction)

    assert max(fractions) - min(fractions) > 0.02


def test_cohort_size_grades_peak_zone_cold_devices():
    peaks = []
    for cohort_size in (1, 2, 4):
        config = _config("cohort", backend="rect")
        config.adoption.initial_adopted_fraction = 0.5
        config.n_steps = 8
        config.cell_size = 1000
        config.adoption.cohort_size = cohort_size
        config.adoption.interval_steps = 10
        model = GarlandModel(config)
        model.run()
        peaks.append(model.metrics.summary()["peak_onboarding_wearables_in_zone"])

    assert peaks == sorted(peaks)
    assert peaks[-1] > peaks[0]


def test_settled_world_can_receive_onboarding_without_fleet_cold_start():
    config = _config("trickle")
    config.adoption.rate = 0.2
    config.baseline_warmup_steps = 5
    model = GarlandModel(config)
    model.run()
    summary = model.metrics.summary()

    assert summary["fleet_cold_start"] is False
    assert summary["post_world_settling_cold_baseline_wearable_step_fraction"] > 0
    assert all(event["step"] >= config.adoption.start_step for event in summary["adoption_events"])


def test_initial_adoption_fraction_leaves_established_population():
    config = _config("trickle")
    config.adoption.initial_adopted_fraction = 0.5
    config.adoption.rate = 0.0
    model = GarlandModel(config)

    assert sum(agent.is_operational for agent in model.citizen_agents) == 20
    assert sum(agent.device_status.name == "NOT_ADOPTED" for agent in model.citizen_agents) == 20


def test_cohort_initial_population_keeps_groups_intact():
    config = _config("cohort")
    config.adoption.initial_adopted_fraction = 0.5
    config.adoption.rate = 0.0
    model = GarlandModel(config)

    adopted_by_household: dict[int, set[bool]] = {}
    for agent in model.citizen_agents:
        adopted_by_household.setdefault(agent.household_id, set()).add(agent.is_operational)
    assert all(len(statuses) == 1 for statuses in adopted_by_household.values())


def test_cohort_adoption_event_covers_complete_household():
    config = _config("cohort", backend="rect")
    config.adoption.initial_adopted_fraction = 0.5
    config.n_steps = 8
    config.cell_size = 1000
    config.adoption.interval_steps = 10
    model = GarlandModel(config)
    model.run()

    adoption_steps = {
        agent.adoption_step
        for agent in model.citizen_agents
        if agent.adoption_step is not None and agent.adoption_step >= 6
    }
    for step in adoption_steps:
        adopted_households = {
            agent.household_id for agent in model.citizen_agents if agent.adoption_step == step
        }
        for household_id in adopted_households:
            assert all(
                agent.adoption_step == step
                for agent in model.citizen_agents
                if agent.household_id == household_id
            )


def test_non_default_full_initial_fraction_is_rejected():
    config = _config("trickle")
    config.adoption.initial_adopted_fraction = 1.0
    with pytest.raises(ValueError, match="initial_adopted_fraction"):
        GarlandModel(config)


def test_onboarding_window_is_separate_from_covariance_prior_state():
    peaks = []
    for window in (2, 10):
        config = _config("trickle")
        config.adoption.onboarding_window_steps = window
        config.adoption.rate = 0.2
        model = GarlandModel(config)
        model.run()
        peaks.append(model.metrics.summary()["peak_onboarding_cold_wearables_in_zone"])

    assert peaks[1] > peaks[0]
