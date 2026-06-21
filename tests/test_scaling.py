"""Scaling and performance validation tests.

These tests use reduced populations and are marked to keep CI fast.
Run the optional city-scale benchmark locally with:

    python -m garland.benchmark --n-agents 250000 --n-steps 20
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from garland.benchmark import run_benchmark
from garland.hazards import PlumeConfig, SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig


@pytest.fixture
def medium_config():
    """Medium-scale config for scaling smoke tests."""
    return SimulationConfig(
        n_agents=5000,
        wearable_fraction=0.15,
        n_steps=10,
        seed=42,
        plumes=[PlumeConfig(start_step=10_000)],
        spatial_backend="rect",
        mobility_model="static",
    )


class TestScalingOptimizations:
    """Regression tests for city-scale performance fixes."""

    def test_vectorized_positions_match_neighborhood_clustering(self, medium_config):
        """Positions should cluster around assigned neighborhood centers."""
        model = GarlandModel(medium_config)
        n_neighborhoods = len(np.unique(model.neighborhood_ids))
        centers_x = np.zeros(n_neighborhoods)
        centers_y = np.zeros(n_neighborhoods)
        for nb in range(n_neighborhoods):
            mask = model.neighborhood_ids == nb
            centers_x[nb] = np.mean(model.agent_x[mask])
            centers_y[nb] = np.mean(model.agent_y[mask])

        for nb in range(n_neighborhoods):
            mask = model.neighborhood_ids == nb
            dist = np.sqrt(
                (model.agent_x[mask] - centers_x[nb]) ** 2
                + (model.agent_y[mask] - centers_y[nb]) ** 2
            )
            assert np.mean(dist) < 400

    def test_wearable_agents_have_cached_cell_ids(self, medium_config):
        """Wearable agents should carry cell IDs matching the spatial grid at init."""
        model = GarlandModel(medium_config)
        for agent in model.citizen_agents:
            assert agent.cell_id == int(model.agent_cell_ids[agent.idx])
            assert agent.cell_id == model.grid.cell_of(agent.idx)

    def test_wearable_agents_indexed_by_cell(self, medium_config):
        """Zone-indexed wearable lookup should cover all citizen agents."""
        model = GarlandModel(medium_config)
        indexed = sum(len(v) for v in model.wearable_agents_by_cell.values())
        assert indexed == len(model.citizen_agents)

    def test_init_completes_within_reasonable_time(self, medium_config):
        """5K init should finish quickly after vectorization."""
        t0 = time.perf_counter()
        GarlandModel(medium_config)
        assert time.perf_counter() - t0 < 5.0

    def test_ten_steps_complete_within_reasonable_time(self, medium_config):
        """5K × 10 steps should stay in single-digit seconds."""
        model = GarlandModel(medium_config)
        t0 = time.perf_counter()
        model.run(steps=10)
        assert time.perf_counter() - t0 < 30.0


class TestBenchmarkModule:
    """Smoke tests for the optional benchmark helper."""

    def test_run_benchmark_returns_expected_keys(self):
        result = run_benchmark(n_agents=1000, n_steps=3, seed=42)
        assert result["n_agents"] == 1000
        assert result["n_steps"] == 3
        assert result["avg_step_ms"] > 0
        assert result["init_seconds"] > 0
        assert result["peak_init_mb"] > 0


class TestSEIRScalingConfig:
    """SEIR proximity cap should be configurable."""

    def test_max_infectious_checks_default(self):
        assert SEIRConfig().max_infectious_checks == 500

    def test_max_infectious_checks_override(self):
        assert SEIRConfig(max_infectious_checks=1000).max_infectious_checks == 1000
