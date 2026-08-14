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
        row["not_adopted_wearables"] + row["adopted_wearables"] == wearable_count
        for row in rows
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
        config = _config("cohort")
        config.adoption.cohort_size = cohort_size
        config.adoption.interval_steps = 10
        model = GarlandModel(config)
        model.run()
        peaks.append(model.metrics.summary()["peak_onboarding_cold_wearables_in_zone"])

    assert peaks == sorted(peaks)
    assert peaks[-1] > peaks[0]


def test_settled_world_can_receive_onboarding_without_fleet_cold_start():
    config = _config("trickle")
    config.adoption.rate = 0.2
    model = GarlandModel(config)
    model.run()
    summary = model.metrics.summary()

    assert summary["fleet_cold_start"] is False
    assert summary["post_world_settling_cold_baseline_wearable_step_fraction"] > 0
    assert all(
        event["step"] >= config.adoption.start_step
        for event in summary["adoption_events"]
    )
