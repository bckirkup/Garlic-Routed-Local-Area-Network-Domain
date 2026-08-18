"""Regression tests for calibrated toxin exposure and footprint reporting."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.calibrate_toxin_footprints import calibrate_rate

from garland.channels import DEFAULT_CHANNEL_SET
from garland.config import config_from_dict, config_to_dict
from garland.hazards import (
    RESPIRATORY_DELTA_ASYMPTOTE_BPM,
    concentration_for_respiratory_delta,
    plume_biometric_perturbation,
    respiratory_delta_for_concentration,
)
from garland.metrics import DetectionEvent, MetricsCollector
from garland.privacy import AnomalyType
from garland.simulation import GarlandModel, SimulationConfig


def test_concentration_gate_is_monotone_and_inverts_dose_response() -> None:
    deltas = (0.5, 2.0, 6.0)
    gates = [concentration_for_respiratory_delta(delta) for delta in deltas]
    assert gates == sorted(gates)
    for delta, gate in zip(deltas, gates):
        assert respiratory_delta_for_concentration(gate) == pytest.approx(delta)


def test_perturbation_remains_continuous_below_evaluation_gate() -> None:
    below_gate = plume_biometric_perturbation(0.05)
    at_gate = plume_biometric_perturbation(concentration_for_respiratory_delta(2.0))
    respiratory_index = DEFAULT_CHANNEL_SET.index("respiratory_rate")
    assert np.any(below_gate)
    assert at_gate[respiratory_index] == pytest.approx(2.0)
    assert at_gate[respiratory_index] > below_gate[respiratory_index]


@pytest.mark.parametrize("delta", (0.0, -1.0, RESPIRATORY_DELTA_ASYMPTOTE_BPM, 13.0))
def test_invalid_respiratory_delta_rejected(delta: float) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(minimum_respiratory_delta_bpm=delta)


def test_legacy_gate_is_explicit() -> None:
    dose_config = SimulationConfig()
    legacy_config = SimulationConfig(toxin_exposure_gate_mode="legacy_0_01")
    assert dose_config.toxin_exposure_concentration_gate() == pytest.approx(0.1)
    assert legacy_config.toxin_exposure_concentration_gate() == pytest.approx(0.01)
    assert dose_config.toxin_physiology_concentration_floor() == pytest.approx(
        concentration_for_respiratory_delta(0.1)
    )
    assert legacy_config.toxin_physiology_concentration_floor() == pytest.approx(0.01)


def test_exposure_gate_config_round_trips() -> None:
    config = SimulationConfig(
        minimum_respiratory_delta_bpm=6.0,
        toxin_exposure_gate_mode="legacy_0_01",
    )
    restored = config_from_dict(config_to_dict(config))
    assert restored.minimum_respiratory_delta_bpm == pytest.approx(6.0)
    assert restored.toxin_exposure_gate_mode == "legacy_0_01"


def test_small_grid_positions_are_in_bounds() -> None:
    model = GarlandModel(
        SimulationConfig(
            n_agents=40,
            grid_width=600.0,
            grid_height=700.0,
            cell_size=100.0,
            spatial_backend="rect",
            n_steps=1,
        )
    )
    assert np.all((model.agent_x >= 0.0) & (model.agent_x <= 600.0))
    assert np.all((model.agent_y >= 0.0) & (model.agent_y <= 700.0))


def test_large_grid_positions_preserve_legacy_seeded_draws() -> None:
    config = SimulationConfig(
        n_agents=40,
        grid_width=2_000.0,
        grid_height=2_000.0,
        cell_size=100.0,
        spatial_backend="rect",
        seed=123,
        n_steps=1,
    )
    model = GarlandModel(config)
    rng = np.random.default_rng(config.seed)
    n_neighborhoods = max(
        1, config.n_agents // (config.households_per_neighborhood * config.household_size_mean)
    )
    centers_x = rng.uniform(500, config.grid_width - 500, n_neighborhoods)
    centers_y = rng.uniform(500, config.grid_height - 500, n_neighborhoods)
    neighborhood_ids = rng.integers(0, n_neighborhoods, config.n_agents)
    household_ids = np.empty(config.n_agents, dtype=np.int64)
    next_household_id = 0
    for neighborhood in range(n_neighborhoods):
        members = np.nonzero(neighborhood_ids == neighborhood)[0]
        rng.shuffle(members)
        for start in range(0, len(members), config.household_size_mean):
            household_ids[members[start : start + config.household_size_mean]] = next_household_id
            next_household_id += 1
    expected_x = np.clip(
        centers_x[neighborhood_ids] + rng.normal(0, 300, config.n_agents),
        0,
        config.grid_width,
    ).astype(np.float32)
    expected_y = np.clip(
        centers_y[neighborhood_ids] + rng.normal(0, 300, config.n_agents),
        0,
        config.grid_height,
    ).astype(np.float32)
    np.testing.assert_array_equal(model.agent_x, expected_x)
    np.testing.assert_array_equal(model.agent_y, expected_y)


def test_dosed_toxin_metrics_are_bounded_and_count_hollow_events() -> None:
    metrics = MetricsCollector()
    for dosed in (0, 1, 2):
        metrics.record_detection(
            DetectionEvent(
                step=dosed,
                hazard_type="toxin",
                anomaly_type=AnomalyType.RESPIRATORY,
                zone_id=0,
                true_positive=True,
                agents_affected=4,
                dosed_agents=dosed,
            )
        )
    summary = metrics.summary()
    assert summary["toxin_true_positive_dosed_agents_evaluation_only"] == [0, 1, 2]
    assert summary["toxin_true_positives_fewer_than_two_dosed_agents_evaluation_only"] == 2
    assert all(
        dosed <= event.agents_affected
        for event in metrics.detection_events
        if event.dosed_agents is not None
        for dosed in (event.dosed_agents,)
    )


def test_calibration_area_and_device_count_increase_with_release_rate() -> None:
    rows = [
        calibrate_rate(rate, "D", grid_m=500.0, sample_step_m=10.0) for rate in (5.0, 50.0, 500.0)
    ]
    areas = [float(row["footprint_area_hectares"]) for row in rows]
    devices = [float(row["implied_device_count"]) for row in rows]
    assert areas == sorted(areas)
    assert devices == sorted(devices)
    for value in areas + devices:
        assert np.isfinite(value)
        assert value >= 0.0
    for row in rows:
        assert row["implied_device_count"] <= row["grid_wearable_population"]
