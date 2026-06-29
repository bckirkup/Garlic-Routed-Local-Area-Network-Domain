"""Tests for wearable device lifecycle (battery, removal, power-off)."""

from __future__ import annotations

import numpy as np
import pytest

from garland.agents import CitizenAgent
from garland.biometrics import generate_profiles
from garland.device_lifecycle import DeviceLifecycleConfig, DeviceLifecycleEngine, DeviceStatus
from garland.hazards import PlumeConfig, SEIRConfig
from garland.privacy import AnomalyType, BroadcastQuery, PrivacyConfig
from garland.simulation import GarlandModel, SimulationConfig


def _lifecycle_config(**kwargs) -> DeviceLifecycleConfig:
    """Build an enabled lifecycle config with optional overrides."""
    defaults = {
        "enabled": True,
        "battery_enabled": True,
        "drain_per_step": 0.01,
        "removal_enabled": False,
        "power_off_enabled": False,
        "home_charge_enabled": False,
    }
    defaults.update(kwargs)
    return DeviceLifecycleConfig(**defaults)


def _sim_config(**kwargs) -> SimulationConfig:
    """Small static simulation config for lifecycle tests."""
    defaults = {
        "n_agents": 200,
        "wearable_fraction": 0.5,
        "grid_width": 2000.0,
        "grid_height": 2000.0,
        "cell_size": 200.0,
        "n_steps": 50,
        "seed": 42,
        "mobility_model": "static",
        "seir": SEIRConfig(initial_infected=0, beta=0.0),
        "plumes": [PlumeConfig(start_step=9999, duration_steps=1)],
    }
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestDeviceLifecycleEngine:
    """Unit tests for DeviceLifecycleEngine transitions."""

    def test_battery_depletes_over_time(self, rng):
        cfg = DeviceLifecycleConfig(
            battery_enabled=True,
            drain_per_step=0.1,
            removal_enabled=False,
            power_off_enabled=False,
            home_charge_enabled=False,
        )
        engine = DeviceLifecycleEngine(10, cfg, rng)
        initial = engine.battery_levels.copy()
        for _ in range(10):
            at_home = np.zeros(10, dtype=bool)
            engine.step(12.0, activity_level=0.3, at_home_mask=at_home, rng=rng)
        assert np.all(engine.battery_levels < initial)
        assert np.any(engine.status == DeviceStatus.DEPLETED)

    def test_home_charging_recovers(self, rng):
        cfg = DeviceLifecycleConfig(
            battery_enabled=True,
            drain_per_step=1.0,
            removal_enabled=False,
            power_off_enabled=False,
            home_charge_enabled=True,
            home_charge_rate=0.5,
            battery_capacity=1.0,
        )
        engine = DeviceLifecycleEngine(5, cfg, rng)
        at_home = np.ones(5, dtype=bool)
        engine.step(23.0, activity_level=0.0, at_home_mask=at_home, rng=rng)
        assert np.all(engine.status == DeviceStatus.DEPLETED)
        for _ in range(5):
            engine.step(23.0, activity_level=0.0, at_home_mask=at_home, rng=rng)
        assert np.all(engine.status == DeviceStatus.ACTIVE)
        assert np.all(engine.battery_levels >= cfg.battery_capacity)


class TestCitizenAgentOperational:
    """Tests for CitizenAgent operability gating."""

    def test_depleted_device_no_tokens(self, rng):
        profile = generate_profiles(1, rng)[0]
        agent = CitizenAgent(
            idx=0,
            has_wearable=True,
            profile=profile,
            device_status=DeviceStatus.DEPLETED,
        )
        token = agent.observe_and_detect(
            hour=12,
            month=1,
            day_of_year=15,
            hour_of_day=12.0,
            rng=rng,
            cell_id=0,
        )
        assert token is None

    def test_power_off_stops_responses(self, rng):
        agent = CitizenAgent(idx=0, has_wearable=True, device_status=DeviceStatus.POWERED_OFF)
        agent.anomaly_active = True
        agent.anomaly_type = AnomalyType.CARDIAC
        query = BroadcastQuery(
            zone_cells=[0],
            anomaly_type=AnomalyType.CARDIAC,
            time_window_start=0,
            time_window_end=12,
            query_id=0,
        )
        resp = agent.respond_to_query(query, 100.0, 200.0, 0, PrivacyConfig(), rng)
        assert resp is None

    def test_offline_clears_anomaly_on_sync(self):
        config = _sim_config(device_lifecycle=_lifecycle_config())
        model = GarlandModel(config)
        agent = model.citizen_agents[0]
        agent.anomaly_active = True
        agent.anomaly_type = AnomalyType.FEBRILE
        assert model.device_lifecycle_engine is not None
        model.device_lifecycle_engine.status[0] = DeviceStatus.NOT_WORN
        model._sync_citizen_device_state()
        assert not agent.anomaly_active
        assert agent.anomaly_type is None


class TestSimulationIntegration:
    """Integration tests through GarlandModel.step()."""

    def test_disabled_preserves_current_behavior(self):
        config = _sim_config(device_lifecycle=DeviceLifecycleConfig(enabled=False))
        model = GarlandModel(config)
        assert model.device_lifecycle_engine is None
        for agent in model.citizen_agents:
            assert agent.device_status == DeviceStatus.ACTIVE
            assert agent.is_operational
        for _ in range(5):
            model.step()
        for agent in model.citizen_agents:
            assert agent.is_operational

    def test_removal_stops_sensing(self, rng):
        config = _sim_config(device_lifecycle=_lifecycle_config())
        model = GarlandModel(config)
        engine = model.device_lifecycle_engine
        assert engine is not None
        engine.status[:] = DeviceStatus.NOT_WORN
        model._sync_citizen_device_state()

        model.step()
        for agent in model.citizen_agents:
            assert not agent.is_operational

    def test_redon_restores_activity(self, rng):
        config = _sim_config(
            device_lifecycle=_lifecycle_config(
                removal_enabled=True,
                redon_prob=1.0,
                removal_prob_wake=0.0,
            )
        )
        model = GarlandModel(config)
        engine = model.device_lifecycle_engine
        assert engine is not None
        engine.status[:] = DeviceStatus.NOT_WORN
        model._sync_citizen_device_state()

        model._update_device_lifecycle(12.0, activity_level=0.2)
        assert np.all(engine.status == DeviceStatus.ACTIVE)

    def test_metrics_track_coverage(self):
        config = _sim_config(
            device_lifecycle=_lifecycle_config(
                drain_per_step=0.05,
                removal_enabled=True,
                removal_prob_sleep=0.5,
                power_off_enabled=True,
                power_off_prob_night=0.5,
            )
        )
        model = GarlandModel(config)
        n_wearable = len(model.citizen_agents)
        for _ in range(30):
            model.step()

        records = model.metrics.step_records
        assert len(records) == 30
        assert "wearables_active" in records[-1]
        assert "wearables_offline" in records[-1]
        assert "mean_battery_level" in records[-1]
        assert records[-1]["wearables_active"] <= n_wearable
        assert records[-1]["wearables_offline"] >= 0

    def test_config_file_loads_device_lifecycle(self):
        from garland.config import load_config_file

        config = load_config_file("examples/device_lifecycle.yaml")
        assert config.device_lifecycle.enabled is True
        assert config.device_lifecycle.drain_per_step == pytest.approx(0.002)
