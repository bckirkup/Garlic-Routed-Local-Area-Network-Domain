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
    operating_rate = model.metrics.summary()["broadcasts_per_1000_agents_per_day"]
    assert operating_rate > 0
    assert operating_rate == pytest.approx(763.3333333333, rel=1e-3)


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


def test_anomaly_threshold_grades_background_rate():
    rates = []
    for threshold in (3.0, 4.5, 6.0):
        config = load_config_file(ROOT / "examples/null_baseline.yaml")
        config.n_agents = 300
        config.n_steps = 288
        config.anomaly_threshold = threshold
        model = GarlandModel(config)
        model.run()
        rates.append(model.metrics.summary()["background_rate"])

    assert rates == sorted(rates, reverse=True)
    assert rates[0] - rates[-1] > 0.001


def test_null_alarm_rate_does_not_grow_monotonically_after_warmup():
    config = load_config_file(ROOT / "examples/null_baseline.yaml")
    config.n_agents = 100
    config.n_steps = 1728
    model = GarlandModel(config)
    model.run()

    daily_token_rates = []
    for day in range(1, 6):
        rows = model.metrics.step_records[day * 288 : (day + 1) * 288]
        active_wearable_steps = sum(int(row["wearables_active"]) for row in rows)
        daily_token_rates.append(
            sum(row["tokens_submitted"] for row in rows) / active_wearable_steps
        )

    # The buggy implementation measured 0.876% -> 1.564% (1.79x);
    # the fixed implementation is about 0.85x, so 1.25 separates them.
    assert daily_token_rates[-1] <= daily_token_rates[0] * 1.25


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


def _background_metrics(
    counts: list[int], *, eligible: int = 10, threshold: int = 5
) -> dict:
    metrics = MetricsCollector()
    metrics.record_aggregation_threshold_config(threshold)
    for zone_id, count in enumerate(counts):
        metrics.record_background_step(
            0,
            0,
            {zone_id: eligible},
            {(zone_id, AnomalyType.RESPIRATORY): count},
        )
    return metrics.summary()


def test_background_dispersion_is_near_one_for_independent_poisson_groups():
    rng = np.random.default_rng(42)
    summary = _background_metrics(rng.poisson(0.5, 400).tolist())

    assert summary["background_group_count"] > 0
    assert summary["background_pearson_dispersion"] == pytest.approx(1.0, abs=0.2)


def test_background_dispersion_detects_and_grades_clustering():
    less_clustered = _background_metrics(([1] * 50) + ([0] * 350))
    more_clustered = _background_metrics(([5] * 10) + ([0] * 390))

    assert less_clustered["background_pearson_dispersion"] >= 0
    assert more_clustered["background_pearson_dispersion"] > (
        less_clustered["background_pearson_dispersion"] + 1.0
    )


def _background_window_metrics(
    counts_by_bin: list[list[int]], *, window_bins: int = 3, threshold: int = 5
) -> dict:
    metrics = MetricsCollector()
    metrics.record_aggregation_threshold_config(threshold)
    metrics.record_aggregation_window_config(window_bins)
    for time_bin, counts in enumerate(counts_by_bin):
        for zone_id, count in enumerate(counts):
            metrics.record_background_step(
                time_bin * 12,
                time_bin,
                {zone_id: 10},
                {(zone_id, AnomalyType.RESPIRATORY): count},
            )
    return metrics.summary()


def test_background_window_dispersion_is_near_one_for_independent_groups():
    rng = np.random.default_rng(42)
    counts = rng.poisson(0.5, (12, 100)).tolist()
    summary = _background_window_metrics(counts)

    assert summary["background_window_group_count"] > 0
    assert summary["background_window_pearson_dispersion"] == pytest.approx(
        1.0, abs=0.25
    )
    assert 0 <= summary["background_window_observed_at_threshold_fraction"] <= 1
    assert 0 <= summary["background_window_poisson_tail_fraction"] <= 1


def test_background_window_dispersion_grades_clustering():
    less_clustered = _background_window_metrics(
        [[1] * 50 + [0] * 50 for _ in range(12)]
    )
    more_clustered = _background_window_metrics(
        [[5] * 10 + [0] * 90 for _ in range(12)]
    )

    assert less_clustered["background_window_pearson_dispersion"] >= 0
    assert more_clustered["background_window_pearson_dispersion"] > (
        less_clustered["background_window_pearson_dispersion"] + 1.0
    )


def test_world_settling_separates_transient_and_settled_stream():
    def measure(transient: int) -> dict:
        metrics = MetricsCollector()
        metrics.record_aggregation_threshold_config(5)
        metrics.record_aggregation_window_config(3)
        metrics.record_world_settling_config(3)
        counts = [transient, transient, transient, 0, 1, 0, 1, 0, 1]
        for step, count in enumerate(counts):
            metrics.record_background_step(
                step,
                step,
                {0: 10},
                {(0, AnomalyType.RESPIRATORY): count},
            )
        return metrics.summary()

    small = measure(1)
    medium = measure(3)
    large = measure(6)
    assert large["background_emission_pearson_dispersion"] > (
        medium["background_emission_pearson_dispersion"]
    )
    assert medium["background_emission_pearson_dispersion"] > (
        small["background_emission_pearson_dispersion"]
    )
    assert medium["background_settled_emission_pearson_dispersion"] == pytest.approx(
        0.5, abs=0.5
    )
    assert medium["background_settled_window_pearson_dispersion"] >= 0


def test_world_settling_zero_reproduces_full_metrics():
    metrics = MetricsCollector()
    metrics.record_aggregation_threshold_config(5)
    metrics.record_aggregation_window_config(3)
    metrics.record_world_settling_config(0)
    for step, count in enumerate([0, 1, 0, 2, 0, 1]):
        metrics.record_background_step(
            step,
            step,
            {0: 10},
            {(0, AnomalyType.RESPIRATORY): count},
        )
    summary = metrics.summary()
    assert summary["background_settled_rate"] == summary["background_rate"]
    assert summary["background_settled_rate_by_anomaly_type"] == summary[
        "background_rate_by_anomaly_type"
    ]
    assert summary["background_settled_emission_pearson_dispersion"] == summary[
        "background_emission_pearson_dispersion"
    ]
    assert summary["background_settled_window_pearson_dispersion"] == summary[
        "background_window_pearson_dispersion"
    ]
    assert summary["background_settled_population_variance_to_mean"] == summary[
        "background_population_variance_to_mean"
    ]


def test_background_tail_fractions_are_undefined_without_aggregation_threshold():
    metrics = MetricsCollector()
    metrics.record_aggregation_window_config(3)
    metrics.record_background_step(
        0,
        0,
        {0: 10},
        {(0, AnomalyType.RESPIRATORY): 1},
    )
    summary = metrics.summary()

    assert summary["background_emission_observed_at_threshold_fraction"] is None
    assert summary["background_emission_poisson_tail_fraction"] is None
    assert summary["background_window_observed_at_threshold_fraction"] is None
    assert summary["background_window_poisson_tail_fraction"] is None
    assert summary["background_settled_emission_observed_at_threshold_fraction"] is None
    assert summary["background_settled_emission_poisson_tail_fraction"] is None
    assert summary["background_settled_window_observed_at_threshold_fraction"] is None
    assert summary["background_settled_window_poisson_tail_fraction"] is None


def test_background_summary_bounds_and_undefined_denominators():
    metrics = MetricsCollector()
    metrics.record_aggregation_threshold_config(5)
    summary = metrics.summary()

    assert summary["background_rate"] is None
    assert summary["background_pearson_dispersion"] is None
    assert summary["background_groups_observed_at_threshold_fraction"] is None
    assert summary["background_groups_poisson_tail_fraction"] is None
    assert summary["background_metrics_daily"] == {}


def test_world_settling_marker_reports_unsettled_short_run_and_past_boundary():
    metrics = MetricsCollector()
    metrics.record_world_settling_config(3)
    for step in range(3):
        metrics.record_step(
            step=step,
            seir_counts={"S": 10},
            plume_exposed=0,
            anomalies_detected=0,
            tokens_submitted=0,
            broadcasts_issued=0,
            responses_received=0,
            cumulative_epsilon=0.0,
            operational_wearables=2,
        )

    short = metrics.summary()
    assert short["world_settling_steps"] == 3
    assert short["world_settling_complete"] is False
    assert short["world_settling_status"] == "not_settled"
    assert short["steps_before_world_settling"] == 3
    assert short["steps_after_world_settling"] == 0
    assert short["world_settling_fraction_of_run"] == pytest.approx(1.0)
    assert short["post_world_settling_cold_baseline_wearable_step_fraction"] is None
    assert short["background_settled_rate"] is None

    metrics.record_step(
        step=3,
        seir_counts={"S": 10},
        plume_exposed=0,
        anomalies_detected=0,
        tokens_submitted=0,
        broadcasts_issued=0,
        responses_received=0,
        cumulative_epsilon=0.0,
        operational_wearables=2,
    )
    settled = metrics.summary()
    assert settled["world_settling_complete"] is True
    assert settled["world_settling_status"] == "settled"
    assert settled["steps_before_world_settling"] == 3
    assert settled["steps_after_world_settling"] == 1
    assert settled["world_settling_fraction_of_run"] == pytest.approx(0.75)
    assert metrics.to_dataframe()["past_world_settling"].tolist() == [False, False, False, True]


def test_cold_baseline_fraction_grades_device_churn():
    base = load_config_file(ROOT / "examples/device_lifecycle.yaml")
    base.n_agents = 100
    base.n_steps = 400
    base.world_settling_steps = 288
    base.baseline_warmup_steps = 24

    churn_model = GarlandModel(base)
    churn_model.run()
    churn_summary = churn_model.metrics.summary()

    no_churn = load_config_file(ROOT / "examples/device_lifecycle.yaml")
    no_churn.n_agents = 100
    no_churn.n_steps = 400
    no_churn.world_settling_steps = 288
    no_churn.baseline_warmup_steps = 24
    no_churn.device_lifecycle.removal_enabled = False
    no_churn.device_lifecycle.power_off_enabled = False
    no_churn.device_lifecycle.battery_enabled = False
    no_churn_model = GarlandModel(no_churn)
    no_churn_model.run()
    no_churn_summary = no_churn_model.metrics.summary()

    churn_fraction = churn_summary["fleet_cold_baseline_wearable_step_fraction"]
    no_churn_fraction = no_churn_summary["fleet_cold_baseline_wearable_step_fraction"]
    assert churn_fraction is not None
    assert no_churn_fraction is not None
    assert churn_fraction > no_churn_fraction >= 0
    assert churn_summary["post_world_settling_cold_baseline_wearable_step_fraction"] == 0
    assert no_churn_summary[
        "post_world_settling_cold_baseline_wearable_step_fraction"
    ] == 0


@pytest.mark.parametrize("backend", ["hex", "rect"])
def test_null_background_summary_works_for_both_spatial_backends(backend: str):
    config = load_config_file(ROOT / "examples/null_baseline.yaml")
    config.n_agents = 500
    config.n_steps = 336
    config.world_settling_steps = 288
    config.spatial_backend = backend
    model = GarlandModel(config)
    model.run()
    summary = model.metrics.summary()

    assert 0 <= summary["background_rate"] <= 1
    assert summary["background_pearson_dispersion"] is not None
    assert np.isfinite(summary["background_pearson_dispersion"])
    assert summary["background_pearson_dispersion"] >= 0
    assert 0 <= summary["background_groups_observed_at_threshold_fraction"] <= 1
    assert 0 <= summary["background_groups_poisson_tail_fraction"] <= 1
    settled_rate = summary["background_settled_rate"]
    assert settled_rate is not None
    assert 0 <= settled_rate <= 1
    settled_emission = summary["background_settled_emission_pearson_dispersion"]
    assert settled_emission is not None
    assert summary["world_settling_status"] == "settled"
    assert summary["world_settling_complete"] is True
    assert (
        summary["steps_before_world_settling"]
        + summary["steps_after_world_settling"]
        == config.n_steps
    )
    assert 0 <= summary["world_settling_fraction_of_run"] <= 1
    local_fraction = summary["post_world_settling_cold_baseline_wearable_step_fraction"]
    assert local_fraction == pytest.approx(0.0)
    assert 0 <= local_fraction <= 1
    assert np.isfinite(settled_emission)
    assert settled_emission >= 0
    settled_window = summary["background_settled_window_pearson_dispersion"]
    assert settled_window is not None
    assert np.isfinite(settled_window)
    assert settled_window >= 0
    settled_emission_tail = summary[
        "background_settled_emission_observed_at_threshold_fraction"
    ]
    assert settled_emission_tail is not None
    assert 0 <= settled_emission_tail <= 1
    settled_window_tail = summary[
        "background_settled_window_observed_at_threshold_fraction"
    ]
    assert settled_window_tail is not None
    assert 0 <= settled_window_tail <= 1
