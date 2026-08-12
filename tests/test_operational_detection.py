"""Regression and sensitivity tests for committed operational scenarios."""

from __future__ import annotations

from pathlib import Path

from garland.config import load_config_file
from garland.simulation import GarlandModel

ROOT = Path(__file__).parents[1]


def test_null_baseline_has_no_hazard_onsets():
    config = load_config_file(ROOT / "examples/null_baseline.yaml")
    config.n_agents = 300
    config.n_steps = 48
    model = GarlandModel(config)
    model.run()

    assert all(row["susceptible"] == config.n_agents for row in model.metrics.step_records)
    assert all(row["exposed"] == 0 for row in model.metrics.step_records)
    assert all(row["infectious"] == 0 for row in model.metrics.step_records)
    assert all(row["recovered"] == 0 for row in model.metrics.step_records)
    assert model.metrics.disease_onset_steps == {}
    assert model.metrics.toxin_onset_steps == {}


def test_null_baseline_pins_nonzero_default_alarm_behavior():
    config = load_config_file(ROOT / "examples/null_baseline.yaml")
    config.n_agents = 300
    config.n_steps = 288
    model = GarlandModel(config)
    model.run()

    assert model.metrics.summary()["total_broadcasts"] > 0


def test_anomaly_threshold_changes_operational_alert_rate():
    low = load_config_file(ROOT / "examples/null_baseline.yaml")
    low.n_agents = 300
    low.n_steps = 288
    low.anomaly_threshold = 3.0
    high = load_config_file(ROOT / "examples/null_baseline.yaml")
    high.n_agents = 300
    high.n_steps = 288
    high.anomaly_threshold = 6.0

    low_model = GarlandModel(low)
    high_model = GarlandModel(high)
    low_model.run()
    high_model.run()

    low_alerts = low_model.metrics.summary()["total_broadcasts"]
    high_alerts = high_model.metrics.summary()["total_broadcasts"]
    assert low_alerts > high_alerts


def test_operational_metrics_include_daily_series():
    config = load_config_file(ROOT / "examples/null_baseline.yaml")
    config.n_agents = 300
    config.n_steps = 288
    model = GarlandModel(config)
    model.run()
    summary = model.metrics.summary()

    assert "0" in summary["operational_metrics_daily"]
    day = summary["operational_metrics_daily"]["0"]
    assert day["broadcasts_per_occupied_zone_per_day"] >= 0
    assert day["broadcasts_per_1000_agents_per_day"] >= 0
    assert 0 <= day["fraction_occupied_zones_alarming"] <= 1
