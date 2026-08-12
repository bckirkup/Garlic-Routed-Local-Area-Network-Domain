"""Regression and sensitivity tests for committed operational scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from garland.biometrics import BaselineTracker
from garland.config import load_config_file
from garland.hazards import plume_biometric_perturbation
from garland.metrics import MetricsCollector
from garland.privacy import AnomalyType, classify_anomaly
from garland.simulation import GarlandModel, SimulationConfig

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

    # Deliberately pin the known-bad but stationary default operating point.
    assert model.metrics.summary()["broadcasts_per_1000_agents_per_day"] == pytest.approx(
        596.6666666667
    )


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


def test_null_alarm_rate_does_not_grow_monotonically_after_warmup():
    config = load_config_file(ROOT / "examples/null_baseline.yaml")
    config.n_agents = 100
    config.n_steps = 1728
    model = GarlandModel(config)
    model.run()

    daily_rates = []
    for day in range(1, 6):
        rows = model.metrics.step_records[day * 288 : (day + 1) * 288]
        daily_rates.append(
            sum(row["tokens_submitted"] for row in rows) / (288 * 15)
        )

    assert daily_rates[-1] <= daily_rates[0] * 2


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


def test_partial_day_headline_metrics_are_undefined():
    metrics = MetricsCollector()
    metrics.record_population_config(100)
    for step in range(48):
        metrics.record_step(
            step=step,
            seir_counts={"S": 100},
            plume_exposed=0,
            anomalies_detected=0,
            tokens_submitted=0,
            broadcasts_issued=2,
            responses_received=0,
            cumulative_epsilon=float(step + 1),
            occupied_zone_ids={1},
            alarming_zone_ids={1},
        )

    summary = metrics.summary()
    assert summary["broadcasts_per_occupied_zone_per_day"] is None
    assert summary["broadcasts_per_1000_agents_per_day"] is None
    assert summary["fraction_occupied_zones_alarming"] is None
    assert summary["epsilon_per_agent_per_day"] == pytest.approx(48 / 100 / (48 / 288))


def test_partial_final_day_does_not_supply_headline_metrics():
    metrics = MetricsCollector()
    metrics.record_population_config(100)
    for step in range(300):
        metrics.record_step(
            step=step,
            seir_counts={"S": 100},
            plume_exposed=0,
            anomalies_detected=0,
            tokens_submitted=0,
            broadcasts_issued=1 if step < 288 else 100,
            responses_received=0,
            cumulative_epsilon=float(step + 1),
            occupied_zone_ids={1},
            alarming_zone_ids={1},
        )

    summary = metrics.summary()
    assert summary["broadcasts_per_occupied_zone_per_day"] == pytest.approx(288.0)
    assert summary["broadcasts_per_1000_agents_per_day"] == pytest.approx(2880.0)
    assert summary["fraction_occupied_zones_alarming"] == pytest.approx(1.0)


@pytest.mark.parametrize("spatial_backend", ["hex", "rect"])
def test_wearable_zone_keys_match_agents_after_mobility(spatial_backend):
    config = SimulationConfig(
        n_agents=200,
        wearable_fraction=0.2,
        n_steps=1,
        seed=42,
        spatial_backend=spatial_backend,
        mobility_model="random_walk",
        mobility_speed_m=200.0,
    )
    model = GarlandModel(config)

    for _ in range(5):
        model._update_mobility()
        expected = {int(agent.cell_id) for agent in model.citizen_agents}
        assert set(model.wearable_agents_by_cell) == expected
        assert all(model.wearable_agents_by_cell[cell_id] for cell_id in expected)


def test_cyclical_profile_zero_initialization_is_neutral():
    tracker = BaselineTracker()
    tracker.ema = np.array([70.0, 40.0, 15.0, 36.8])

    np.testing.assert_allclose(
        tracker.expected_baseline(hour=12, month=1),
        tracker.ema,
    )


def test_covariance_uses_pre_update_expected_residual():
    tracker = BaselineTracker()
    tracker.ema = np.array([70.0, 40.0, 15.0, 36.8])
    observation = tracker.ema + np.array([1.0, -2.0, 3.0, 0.5])

    tracker.update(observation, hour=12, month=1)

    expected_residual = observation - np.array([70.0, 40.0, 15.0, 36.8])
    np.testing.assert_allclose(
        tracker.cov_sum,
        np.outer(expected_residual, expected_residual),
    )


def test_toxin_and_infection_patterns_remain_distinguishable():
    toxin = plume_biometric_perturbation(concentration=0.5)
    infection = np.array([15.0, -15.0, 5.0, 1.5])

    assert classify_anomaly(toxin, np.zeros(4)) == AnomalyType.RESPIRATORY
    assert classify_anomaly(infection, np.zeros(4)) in {
        AnomalyType.FEBRILE,
        AnomalyType.MULTI_SYSTEM,
    }
