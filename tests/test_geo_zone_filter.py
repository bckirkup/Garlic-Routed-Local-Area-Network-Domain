"""Planar Laplace displacement must be able to cost detection evidence (#75).

With ``privacy.geo_zone_filter`` enabled the aggregator keeps only replies
whose perturbed reported position falls inside the queried zone, so
``laplace_scale`` becomes a utility knob and not just a budget line. The
default leaves the legacy behaviour untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.agents import CitizenAgent
from garland.config import config_from_dict, config_to_dict
from garland.hazards import PlumeConfig, SEIRConfig, compute_plume_concentration
from garland.privacy import AnomalyType, BroadcastQuery, PerturbedResponse, PrivacyConfig
from garland.simulation import GarlandModel, SimulationConfig
from garland.spatial import H3HexGrid, RectangularGrid, create_spatial_grid

BACKENDS = ["rect", "hex"]
SCALES = [1.0, 50.0, 200.0, 2000.0]
N_REPLIES = 400


def _plume() -> PlumeConfig:
    return PlumeConfig(
        source_x=500.0, source_y=500.0, release_rate=200.0, wind_direction=0.0, start_step=0
    )


def _lone_model(backend: str, **privacy) -> GarlandModel:
    config = SimulationConfig(
        n_agents=1,
        wearable_fraction=1.0,
        grid_width=2000.0,
        grid_height=2000.0,
        cell_size=200.0,
        n_steps=1,
        seed=42,
        spatial_backend=backend,
        plumes=[_plume()],
        seir=SEIRConfig(initial_infected=0),
        privacy=PrivacyConfig(response_mechanism="randomized_response", **privacy),
    )
    model = GarlandModel(config)
    model.agent_x = np.array([700.0], dtype=np.float32)
    model.agent_y = np.array([500.0], dtype=np.float32)
    model.grid.assign_positions(model.agent_x, model.agent_y)
    model.current_step = 5
    return model


def _single_cell_query(model: GarlandModel) -> BroadcastQuery:
    return BroadcastQuery(
        zone_cells=[model.grid.cell_of(0)],
        anomaly_type=AnomalyType.RESPIRATORY,
        time_window_start=0,
        time_window_end=1,
    )


def _matching_agent(model: GarlandModel) -> CitizenAgent:
    agent = model.citizen_agents[0]
    agent.cell_id = model.grid.cell_of(0)
    assert agent.is_operational
    agent.anomaly_active = True
    agent.anomaly_type = AnomalyType.RESPIRATORY
    return agent


def _replies(model: GarlandModel, query: BroadcastQuery, n: int) -> list[PerturbedResponse]:
    """Sample ``n`` genuine replies from the lone agent standing in the zone."""
    agent = _matching_agent(model)
    config = PrivacyConfig(
        response_mechanism="randomized_response",
        randomized_response_p=1.0,
        laplace_scale=model.config.privacy.laplace_scale,
        dummy_rate=0.0,
    )
    rng = np.random.default_rng(11)
    out = []
    for _ in range(n):
        response = agent.respond_to_query(query, 700.0, 500.0, agent.cell_id, config, rng)
        assert response is not None
        out.append(response)
    return out


class TestCellForPosition:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_matches_bulk_assignment(self, backend):
        grid = create_spatial_grid(2000.0, 2000.0, 200.0, backend=backend)
        rng = np.random.default_rng(3)
        x = rng.uniform(0.0, 2000.0, 300).astype(np.float32)
        y = rng.uniform(0.0, 2000.0, 300).astype(np.float32)
        grid.assign_positions(x, y)
        for idx in range(len(x)):
            assert grid.cell_for_position(float(x[idx]), float(y[idx])) == grid.cell_of(idx)

    def test_rect_moves_cell_across_boundary(self):
        grid = RectangularGrid(2000.0, 2000.0, 200.0)
        assert grid.cell_for_position(199.0, 50.0) == 0
        assert grid.cell_for_position(201.0, 50.0) == 1
        assert grid.cell_for_position(50.0, 201.0) == grid.cols

    def test_hex_far_points_land_in_distinct_cells(self):
        grid = H3HexGrid(2000.0, 2000.0, 200.0)
        near = grid.cell_for_position(500.0, 500.0)
        assert grid.cell_for_position(500.5, 500.5) == near
        assert grid.cell_for_position(1500.0, 1500.0) != near


class TestFilterSensitivity:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_kept_fraction_falls_monotonically_with_laplace_scale(self, backend):
        kept = []
        for scale in SCALES:
            model = _lone_model(backend, laplace_scale=scale, geo_zone_filter=True)
            query = _single_cell_query(model)
            replies = _replies(model, query, N_REPLIES)
            filtered = model._filter_responses_to_zone(query, replies)
            kept.append(len(filtered) / N_REPLIES)
            assert model.metrics.geo_zone_filtered_responses == N_REPLIES - len(filtered)
        assert kept[0] > 0.95, kept
        assert kept[-1] < 0.05, kept
        assert all(a > b for a, b in zip(kept, kept[1:])), kept
        assert kept[0] - kept[-1] > 0.9, kept

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_filter_disabled_keeps_every_reply(self, backend):
        model = _lone_model(backend, laplace_scale=2000.0)
        query = _single_cell_query(model)
        replies = _replies(model, query, N_REPLIES)
        assert model._filter_responses_to_zone(query, replies) is replies
        assert model.metrics.geo_zone_filtered_responses == 0


class TestDetectionOutcome:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_large_scale_removes_a_true_positive_small_scale_keeps_it(self, backend):
        """A single genuine reply in a plume is a TP unless displaced out of zone."""
        outcomes = {}
        for scale in (1.0, 20_000.0):
            model = _lone_model(backend, laplace_scale=scale, geo_zone_filter=True)
            query = _single_cell_query(model)
            replies = _replies(model, query, 1)
            filtered = model._filter_responses_to_zone(query, replies)
            concentrations = compute_plume_concentration(
                model.agent_x, model.agent_y, model.plume_config, model.current_step
            )
            model._classify_detection(query, filtered, concentrations)
            outcomes[scale] = len(model.metrics.detection_events)
        assert outcomes == {1.0: 1, 20_000.0: 0}


def _run_config(backend: str, **privacy) -> SimulationConfig:
    return SimulationConfig(
        n_agents=400,
        wearable_fraction=0.5,
        grid_width=1000.0,
        grid_height=1000.0,
        cell_size=200.0,
        n_steps=24,
        seed=7,
        spatial_backend=backend,
        plumes=[PlumeConfig(source_x=500.0, source_y=500.0, release_rate=400.0, start_step=2)],
        seir=SEIRConfig(initial_infected=0),
        privacy=PrivacyConfig(threshold_m=1, k_min=5, **privacy),
    )


def _detection_fingerprint(model: GarlandModel) -> list[tuple[int, str, bool, int]]:
    return [
        (event.step, event.hazard_type, event.true_positive, event.agents_affected)
        for event in model.metrics.detection_events
    ]


class TestFullRun:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_default_off_is_insensitive_to_laplace_scale(self, backend):
        """Negative control: without the filter geo noise still changes nothing."""
        fingerprints = set()
        for scale in (1.0, 200.0, 5000.0):
            model = GarlandModel(_run_config(backend, laplace_scale=scale))
            model.run(steps=24)
            assert model.metrics.geo_zone_filtered_responses == 0
            fingerprints.add(tuple(_detection_fingerprint(model)))
        assert len(fingerprints) == 1

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_filter_on_changes_detection_outcome_with_scale(self, backend):
        dropped = []
        fingerprints = []
        for scale in (1.0, 5000.0):
            model = GarlandModel(_run_config(backend, laplace_scale=scale, geo_zone_filter=True))
            model.run(steps=24)
            dropped.append(model.metrics.geo_zone_filtered_responses)
            fingerprints.append(_detection_fingerprint(model))
            assert model.metrics.total_responses > 0
        assert dropped[0] < dropped[1]
        assert fingerprints[0] != fingerprints[1]


def test_config_round_trip_preserves_flag():
    for value in (False, True):
        config = _run_config("rect", geo_zone_filter=value)
        assert config_from_dict(config_to_dict(config)).privacy.geo_zone_filter is value
