"""Tests for baseline warm-up (cold-start acclimation)."""

from __future__ import annotations

import numpy as np

from garland.agents import CitizenAgent
from garland.biometrics import generate_profiles
from garland.hazards import PlumeConfig, SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig


def _warmup_config(**kwargs) -> SimulationConfig:
    defaults = {
        "n_agents": 1000,
        "wearable_fraction": 0.2,
        "grid_width": 2000.0,
        "grid_height": 2000.0,
        "n_steps": 30,
        "seed": 52,
        "baseline_warmup_steps": 12,
        "seir": SEIRConfig(initial_infected=5, beta=0.03),
        "plumes": [PlumeConfig(start_step=10_000)],
    }
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


class TestBaselineWarmup:
    def test_no_tokens_during_warmup(self):
        model = GarlandModel(_warmup_config())
        for _ in range(model.config.baseline_warmup_steps):
            model.step()
        warmup_records = model.metrics.step_records[: model.config.baseline_warmup_steps]
        assert all(record["tokens_submitted"] == 0 for record in warmup_records)
        assert all(record["baseline_warmup_active"] for record in warmup_records)

    def test_baselines_still_update_during_warmup(self):
        model = GarlandModel(_warmup_config(baseline_warmup_steps=5))
        agent = model.citizen_agents[0]
        samples_before = agent.baseline.n_samples
        model.step()
        assert agent.baseline.n_samples > samples_before

    def test_seir_advances_during_warmup(self):
        model = GarlandModel(_warmup_config(baseline_warmup_steps=20))
        s_before = int(np.sum(model.seir.states == 0))
        for _ in range(5):
            model.step()
        s_after = int(np.sum(model.seir.states == 0))
        assert s_after <= s_before

    def test_detection_resumes_after_warmup(self, monkeypatch):
        """Force an anomaly after warm-up and confirm a token is emitted."""
        model = GarlandModel(_warmup_config(baseline_warmup_steps=3, n_steps=10))
        for agent in model.citizen_agents:
            agent.baseline_warmup_remaining = 0

        agent = model.citizen_agents[0]

        def _always_anomaly(*_args, **_kwargs):
            agent.baseline.update(agent.last_observation, 12, 1)
            from garland.privacy import AnomalyType, EncryptedToken

            return EncryptedToken(
                zone_id=agent.cell_id,
                anomaly_type=AnomalyType.FEBRILE,
                timestamp_bin=0,
                agent_id_hash=1,
            )

        monkeypatch.setattr(agent, "observe_and_detect", _always_anomaly)
        model.step()
        assert model.metrics.step_records[-1]["tokens_submitted"] > 0

    def test_device_adopt_restarts_warmup(self):
        model = GarlandModel(_warmup_config(baseline_warmup_steps=8))
        from garland.device_lifecycle import DeviceLifecycleEngine, DeviceStatus

        model.device_lifecycle_engine = DeviceLifecycleEngine(
            len(model.citizen_agents), model.config.device_lifecycle, model.rng
        )
        agent = model.citizen_agents[0]
        agent.baseline_warmup_remaining = 0
        agent.device_status = DeviceStatus.NOT_WORN
        model.device_lifecycle_engine.status[0] = DeviceStatus.ACTIVE
        model._sync_citizen_device_state()
        assert agent.baseline_warmup_remaining == 8
        assert agent.device_status == DeviceStatus.ACTIVE

    def test_observe_and_detect_suppresses_token(self):
        rng = np.random.default_rng(1)
        profile = generate_profiles(1, rng)[0]
        agent = CitizenAgent(idx=0, has_wearable=True, profile=profile)
        token = agent.observe_and_detect(
            hour=10,
            month=1,
            day_of_year=15,
            hour_of_day=10.0,
            rng=rng,
            cell_id=3,
            suppress_token_emission=True,
        )
        assert token is None
        assert not agent.anomaly_active
        assert agent.baseline.n_samples > 1

    def test_zero_warmup_preserves_legacy_behavior(self):
        model = GarlandModel(_warmup_config(baseline_warmup_steps=0))
        model.run(steps=5)
        assert model.metrics.warmup_step_count() == 0

    def test_config_round_trip(self):
        from garland.config import config_from_dict, config_to_dict

        original = _warmup_config(warmup_on_device_adopt=False)
        restored = config_from_dict(config_to_dict(original))
        assert restored.baseline_warmup_steps == 12
        assert restored.warmup_on_device_adopt is False
