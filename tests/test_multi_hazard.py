"""Tests for multi-plume and multi-outbreak hazard scenarios."""

from __future__ import annotations

import numpy as np

from garland.config import config_from_dict, load_config_file
from garland.hazards import (
    OutbreakSeed,
    PlumeConfig,
    SEIRConfig,
    SEIREngine,
    SEIRState,
    compute_plume_concentrations,
)
from garland.simulation import GarlandModel, SimulationConfig


class TestMultiPlume:
    def test_compute_plume_concentrations_sums_sources(self):
        x = np.array([5100.0, 5100.0, 4900.0], dtype=np.float32)
        y = np.array([5000.0, 5000.0, 5000.0], dtype=np.float32)
        plumes = [
            PlumeConfig(
                plume_id="a",
                source_x=5000.0,
                source_y=5000.0,
                start_step=0,
                duration_steps=100,
            ),
            PlumeConfig(
                plume_id="b",
                source_x=5000.0,
                source_y=5000.0,
                release_rate=2.0,
                start_step=0,
                duration_steps=100,
            ),
        ]
        total, per_plume = compute_plume_concentrations(x, y, plumes, current_step=10)
        assert set(per_plume) == {"a", "b"}
        np.testing.assert_allclose(total, per_plume["a"] + per_plume["b"])

    def test_independent_plume_timing(self):
        x = np.array([5100.0], dtype=np.float32)
        y = np.array([5000.0], dtype=np.float32)
        early = PlumeConfig(plume_id="early", start_step=0, duration_steps=10)
        late = PlumeConfig(plume_id="late", start_step=50, duration_steps=10)
        _, at_5 = compute_plume_concentrations(x, y, [early, late], current_step=5)
        _, at_55 = compute_plume_concentrations(x, y, [early, late], current_step=55)
        assert at_5["early"][0] > 0.0
        assert at_5["late"][0] == 0.0
        assert at_55["early"][0] == 0.0
        assert at_55["late"][0] > 0.0


class TestMultiOutbreak:
    def test_timed_outbreak_seeding(self):
        n = 500
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 4000, n).astype(np.float32)
        y = rng.uniform(0, 4000, n).astype(np.float32)
        config = SEIRConfig(
            initial_infected=0,
            outbreaks=[
                OutbreakSeed(
                    outbreak_id="wave_a",
                    start_step=10,
                    initial_infected=4,
                    center_x=500.0,
                    center_y=500.0,
                    seed_radius=300.0,
                )
            ],
        )
        engine = SEIREngine(config=config)
        engine.initialize(n, rng, x, y)
        assert int(np.sum(engine.states == SEIRState.INFECTIOUS)) == 0
        engine.maybe_seed_outbreaks(10, x, y, rng)
        assert int(np.sum(engine.states == SEIRState.INFECTIOUS)) == 4
        assert np.all(engine.outbreak_origin[engine.states == SEIRState.INFECTIOUS] == "wave_a")

    def test_outbreak_origin_propagates_on_transmission(self):
        n = 20
        rng = np.random.default_rng(1)
        x = np.linspace(0, 40, n, dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        config = SEIRConfig(beta=1.0, initial_infected=0, contact_radius=5.0)
        engine = SEIREngine(config=config)
        engine.initialize(n, rng, x, y)
        engine.states[0] = SEIRState.INFECTIOUS
        engine.outbreak_origin[0] = "wave_x"
        engine.step(0, x, y, rng)
        exposed = engine.states == SEIRState.EXPOSED
        assert np.any(exposed)
        assert np.all(engine.outbreak_origin[exposed] == "wave_x")


class TestMultiHazardConfig:
    def test_config_parses_plume_list(self):
        config = config_from_dict(
            {
                "plumes": [
                    {"plume_id": "a", "start_step": 1},
                    {"plume_id": "b", "start_step": 2},
                ]
            }
        )
        assert len(config.plumes) == 2
        assert config.plumes[0].plume_id == "a"
        assert config.plume.plume_id == "a"

    def test_legacy_single_plume_dict(self):
        config = config_from_dict({"plume": {"start_step": 12, "source_x": 100.0}})
        assert len(config.plumes) == 1
        assert config.plumes[0].start_step == 12

    def test_outbreak_list_in_seir(self):
        config = config_from_dict(
            {
                "seir": {
                    "outbreaks": [
                        {"outbreak_id": "w1", "start_step": 5, "initial_infected": 3}
                    ]
                }
            }
        )
        assert len(config.seir.outbreaks) == 1
        assert config.seir.outbreaks[0].outbreak_id == "w1"

    def test_example_multi_hazard_yaml(self):
        config = load_config_file("examples/multi_hazard.yaml")
        assert len(config.plumes) == 2
        assert len(config.seir.outbreaks) == 2


class TestMultiHazardSimulation:
    def test_simulation_runs_with_multi_hazards(self):
        config = SimulationConfig(
            n_agents=80,
            n_steps=60,
            seed=3,
            grid_width=4000.0,
            grid_height=4000.0,
            mobility_model="static",
            spatial_backend="rect",
            seir=SEIRConfig(
                initial_infected=0,
                outbreaks=[
                    OutbreakSeed(
                        outbreak_id="wave_a",
                        start_step=5,
                        initial_infected=4,
                        center_x=1000.0,
                        center_y=1000.0,
                        seed_radius=400.0,
                    ),
                    OutbreakSeed(
                        outbreak_id="wave_b",
                        start_step=35,
                        initial_infected=4,
                        center_x=1000.0,
                        center_y=1000.0,
                        seed_radius=400.0,
                    ),
                ],
            ),
            plumes=[
                PlumeConfig(
                    plume_id="leak_n",
                    source_x=500.0,
                    source_y=1000.0,
                    wind_direction=0.0,
                    release_rate=20.0,
                    start_step=15,
                    duration_steps=20,
                ),
                PlumeConfig(
                    plume_id="leak_e",
                    source_x=500.0,
                    source_y=1000.0,
                    wind_direction=0.0,
                    release_rate=20.0,
                    start_step=40,
                    duration_steps=15,
                ),
            ],
        )
        model = GarlandModel(config)
        n = config.n_agents
        model.agent_x = np.linspace(700, 1300, n, dtype=np.float32)
        model.agent_y = np.full(n, 1000.0, dtype=np.float32)
        model.grid.assign_positions(model.agent_x, model.agent_y)
        metrics = model.run()
        assert metrics.toxin_onset_steps.get("leak_n") is not None
        assert metrics.toxin_onset_steps.get("leak_e") is not None
        assert metrics.disease_onset_steps.get("wave_a") is not None
        assert metrics.disease_onset_steps.get("wave_b") is not None
