"""Tests for agent mobility and dynamic cell membership."""

from __future__ import annotations

import numpy as np

from garland.hazards import PlumeConfig
from garland.simulation import GarlandModel, SimulationConfig


def _mobility_config(**kwargs) -> SimulationConfig:
    defaults = dict(
        n_agents=500,
        wearable_fraction=0.2,
        grid_width=2000.0,
        grid_height=2000.0,
        n_steps=5,
        seed=7,
        plumes=[PlumeConfig(start_step=10_000)],
        mobility_model="random_walk",
        mobility_speed_m=200.0,
    )
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


class TestMobility:
    def test_random_walk_changes_positions(self):
        model = GarlandModel(_mobility_config())
        x_before = model.agent_x.copy()
        y_before = model.agent_y.copy()
        model._update_mobility()
        moved = np.sum((model.agent_x != x_before) | (model.agent_y != y_before))
        assert moved > 0

    def test_static_mobility_leaves_positions_unchanged(self):
        model = GarlandModel(_mobility_config(mobility_model="static"))
        x_before = model.agent_x.copy()
        y_before = model.agent_y.copy()
        model._update_mobility()
        assert np.array_equal(model.agent_x, x_before)
        assert np.array_equal(model.agent_y, y_before)

    def test_cell_membership_tracks_movement(self):
        model = GarlandModel(_mobility_config())
        initial_cells = model.agent_cell_ids.copy()
        for _ in range(5):
            model._update_mobility()
        assert not np.array_equal(model.agent_cell_ids, initial_cells)

    def test_wearable_cell_ids_stay_consistent_after_moves(self):
        model = GarlandModel(_mobility_config())
        for _ in range(3):
            model._update_mobility()
        for agent in model.citizen_agents:
            assert agent.cell_id == int(model.agent_cell_ids[agent.idx])
            assert agent.cell_id == model.grid.cell_of(agent.idx)

    def test_wearable_index_covers_all_agents_after_moves(self):
        model = GarlandModel(_mobility_config())
        for _ in range(3):
            model._update_mobility()
        indexed = sum(len(v) for v in model.wearable_agents_by_cell.values())
        assert indexed == len(model.citizen_agents)

    def test_simulation_step_with_mobility_runs(self):
        model = GarlandModel(_mobility_config())
        model.run(steps=5)
        assert model.current_step == 5
