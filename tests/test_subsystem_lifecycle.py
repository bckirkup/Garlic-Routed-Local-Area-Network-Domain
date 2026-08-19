"""Tests for independent per-subsystem batteries and lifecycle state.

Each adopted sensor subsystem is its own hardware with its own cell, so the
behaviour under test is *independence*: one subsystem running down, coming off,
or charging must not move any other subsystem's channels.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from garland.device_lifecycle import (
    DeviceLifecycleConfig,
    DeviceLifecycleEngine,
    DeviceStatus,
    SubsystemLifecycle,
    subsystem_config,
)
from garland.devices import (
    ABDOMINAL_ACOUSTIC_BAND,
    BASE_DEVICE_KIND,
    THORACIC_EIT_ACOUSTIC_BAND,
    DeviceFleet,
    DeviceFleetConfig,
    DeviceKind,
    SubsystemPowerProfile,
)
from garland.simulation import GarlandModel, SimulationConfig

ALL_MODALITIES = {
    THORACIC_EIT_ACOUSTIC_BAND.name: 0.6,
    ABDOMINAL_ACOUSTIC_BAND.name: 0.4,
}

CORE_CHANNEL = "heart_rate"
THORACIC_CHANNEL = "regional_ventilation_heterogeneity"
ABDOMINAL_CHANNEL = "bowel_sound_burst_rate"


def make_fleet(n_wearable: int = 300, seed: int = 5) -> DeviceFleet:
    config = DeviceFleetConfig(enabled=True, adoption=dict(ALL_MODALITIES))
    return DeviceFleet(n_wearable, config, np.random.default_rng(seed))


def make_lifecycle(
    fleet: DeviceFleet,
    config: DeviceLifecycleConfig | None = None,
    seed: int = 7,
) -> SubsystemLifecycle:
    return SubsystemLifecycle(
        fleet,
        config if config is not None else DeviceLifecycleConfig(enabled=True),
        np.random.default_rng(seed),
    )


def steps_to_depletion(config: DeviceLifecycleConfig, activity_level: float = 0.0) -> int:
    """Steps a never-charged, never-removed device survives under constant load."""
    quiet = DeviceLifecycleConfig(
        enabled=True,
        battery_capacity=config.battery_capacity,
        drain_per_step=config.drain_per_step,
        activity_drain_multiplier=config.activity_drain_multiplier,
        home_charge_enabled=False,
        removal_enabled=False,
        power_off_enabled=False,
    )
    engine = DeviceLifecycleEngine(1, quiet, np.random.default_rng(0))
    at_home = np.zeros(1, dtype=bool)
    for step in range(1, 200_000):
        engine.step(12.0, activity_level, at_home)
        if engine.status[0] == DeviceStatus.DEPLETED:
            return step
    raise AssertionError("device never depleted")


def test_each_subsystem_gets_its_own_battery_array() -> None:
    lifecycle = make_lifecycle(make_fleet())
    assert set(lifecycle.engines) == set(ALL_MODALITIES)
    thoracic = lifecycle.engines[THORACIC_EIT_ACOUSTIC_BAND.name]
    abdominal = lifecycle.engines[ABDOMINAL_ACOUSTIC_BAND.name]
    assert thoracic.battery_levels is not abdominal.battery_levels
    thoracic.battery_levels[:] = 0.25
    assert np.allclose(abdominal.battery_levels, abdominal.config.battery_capacity)


def test_higher_draw_subsystems_deplete_sooner_but_all_survive_a_while() -> None:
    """Graded power budgets: EIT band < acoustic band < watch in endurance."""
    base = DeviceLifecycleConfig(enabled=True, drain_per_step=0.002)
    lifetimes = {
        kind.name: steps_to_depletion(subsystem_config(base, kind))
        for kind in (BASE_DEVICE_KIND, ABDOMINAL_ACOUSTIC_BAND, THORACIC_EIT_ACOUSTIC_BAND)
    }
    assert (
        lifetimes[THORACIC_EIT_ACOUSTIC_BAND.name]
        < lifetimes[ABDOMINAL_ACOUSTIC_BAND.name]
        < lifetimes[BASE_DEVICE_KIND.name]
    )
    assert all(lifetime > 1 for lifetime in lifetimes.values())


def _kind_with_power(power: SubsystemPowerProfile) -> DeviceKind:
    """A stand-in device kind differing from the thoracic band only in power."""
    return replace(THORACIC_EIT_ACOUSTIC_BAND, power=power)


def test_drain_and_capacity_scaling_moves_endurance_in_the_right_direction() -> None:
    base = DeviceLifecycleConfig(enabled=True, drain_per_step=0.002)
    reference = steps_to_depletion(base)
    thirstier = steps_to_depletion(
        subsystem_config(base, _kind_with_power(SubsystemPowerProfile(drain_multiplier=4.0)))
    )
    bigger_cell = steps_to_depletion(
        subsystem_config(base, _kind_with_power(SubsystemPowerProfile(capacity_multiplier=2.0)))
    )
    assert thirstier < reference < bigger_cell


def test_activity_scaling_is_applied_per_subsystem() -> None:
    base = DeviceLifecycleConfig(enabled=True, drain_per_step=0.002)
    thoracic = subsystem_config(base, THORACIC_EIT_ACOUSTIC_BAND)
    assert thoracic.activity_drain_multiplier > base.activity_drain_multiplier
    moving = steps_to_depletion(thoracic, activity_level=1.0)
    still = steps_to_depletion(thoracic, activity_level=0.0)
    assert moving < still


def test_a_flat_thoracic_band_leaves_the_other_subsystems_reporting() -> None:
    fleet = make_fleet()
    lifecycle = make_lifecycle(fleet)
    thoracic_position = fleet.kinds.index(THORACIC_EIT_ACOUSTIC_BAND)
    abdominal_position = fleet.kinds.index(ABDOMINAL_ACOUSTIC_BAND)
    thoracic_owners = fleet.ownership[:, thoracic_position]
    both = thoracic_owners & fleet.ownership[:, abdominal_position]
    assert both.any(), "fixture needs owners of both bands"

    engine = lifecycle.engines[THORACIC_EIT_ACOUSTIC_BAND.name]
    engine.battery_levels[thoracic_owners] = 0.0
    engine.status[thoracic_owners] = DeviceStatus.DEPLETED

    rng = np.random.default_rng(2)
    core = fleet.channel_set.index(CORE_CHANNEL)
    thoracic = fleet.channel_set.index(THORACIC_CHANNEL)
    abdominal = fleet.channel_set.index(ABDOMINAL_CHANNEL)
    core_seen = 0
    abdominal_seen = 0
    for _ in range(40):
        observed = fleet.observed_matrix(2.0, 0.0, rng, lifecycle.active_matrix())
        assert not observed[:, thoracic].any()
        core_seen += int(np.count_nonzero(observed[both, core]))
        abdominal_seen += int(np.count_nonzero(observed[both, abdominal]))
    assert core_seen > 0
    assert abdominal_seen > 0


def test_a_flat_abdominal_band_leaves_the_thoracic_band_reporting() -> None:
    fleet = make_fleet()
    lifecycle = make_lifecycle(fleet)
    abdominal_position = fleet.kinds.index(ABDOMINAL_ACOUSTIC_BAND)
    owners = fleet.ownership[:, abdominal_position]
    engine = lifecycle.engines[ABDOMINAL_ACOUSTIC_BAND.name]
    engine.status[owners] = DeviceStatus.NOT_WORN

    rng = np.random.default_rng(3)
    observed = fleet.observed_matrix(2.0, 0.0, rng, lifecycle.active_matrix())
    assert not observed[:, fleet.channel_set.index(ABDOMINAL_CHANNEL)].any()
    assert observed[:, fleet.channel_set.index(THORACIC_CHANNEL)].any()


def test_charging_recovers_only_the_depleted_subsystem() -> None:
    fleet = make_fleet(120)
    config = DeviceLifecycleConfig(
        enabled=True,
        drain_per_step=0.0,
        home_charge_rate=0.5,
        removal_enabled=False,
        power_off_enabled=False,
    )
    lifecycle = make_lifecycle(fleet, config)
    thoracic = lifecycle.engines[THORACIC_EIT_ACOUSTIC_BAND.name]
    abdominal = lifecycle.engines[ABDOMINAL_ACOUSTIC_BAND.name]
    owners = fleet.ownership[:, fleet.kinds.index(THORACIC_EIT_ACOUSTIC_BAND)]
    thoracic.battery_levels[owners] = 0.0
    thoracic.status[owners] = DeviceStatus.DEPLETED
    abdominal_before = abdominal.status.copy()

    at_home = np.ones(fleet.n_wearable, dtype=bool)
    for _ in range(6):
        lifecycle.step(2.0, 0.0, at_home)

    assert np.all(thoracic.status[owners] == DeviceStatus.ACTIVE)
    assert np.all(thoracic.battery_levels[owners] > 0.0)
    assert np.array_equal(abdominal.status, abdominal_before)


def test_non_owners_stay_not_adopted_and_never_report() -> None:
    fleet = make_fleet()
    lifecycle = make_lifecycle(fleet)
    at_home = np.ones(fleet.n_wearable, dtype=bool)
    for kind_name, engine in lifecycle.engines.items():
        position = next(i for i, kind in enumerate(fleet.kinds) if kind.name == kind_name)
        non_owners = ~fleet.ownership[:, position]
        assert non_owners.any()
        for _ in range(20):
            lifecycle.step(23.0, 0.5, at_home)
        assert np.all(engine.status[non_owners] == DeviceStatus.NOT_ADOPTED)
        assert not lifecycle.active_matrix()[non_owners, position].any()


def test_lifecycle_state_stays_bounded_and_finite_over_a_long_run() -> None:
    fleet = make_fleet(150)
    lifecycle = make_lifecycle(fleet, DeviceLifecycleConfig(enabled=True, drain_per_step=0.01))
    rng = np.random.default_rng(4)
    valid = {int(status) for status in DeviceStatus}
    for step in range(600):
        hour = (step * 24.0 / 288.0) % 24.0
        lifecycle.step(hour, float(rng.random()), rng.random(fleet.n_wearable) < 0.4)
        for engine in lifecycle.engines.values():
            assert np.all(np.isfinite(engine.battery_levels))
            assert engine.battery_levels.min() >= 0.0
            assert engine.battery_levels.max() <= engine.config.battery_capacity
            assert set(np.unique(engine.status).tolist()) <= valid


def test_subsystem_active_shape_is_validated() -> None:
    fleet = make_fleet(20)
    with pytest.raises(ValueError, match="subsystem_active"):
        fleet.observed_matrix(
            12.0,
            0.0,
            np.random.default_rng(0),
            np.ones((20, len(fleet.kinds) + 1), dtype=np.bool_),
        )


def make_model(seed: int = 11, backend: str = "hex") -> GarlandModel:
    return GarlandModel(
        SimulationConfig(
            n_agents=400,
            n_steps=20,
            wearable_fraction=0.4,
            seed=seed,
            spatial_backend=backend,
            devices=DeviceFleetConfig(enabled=True, adoption=dict(ALL_MODALITIES)),
            device_lifecycle=DeviceLifecycleConfig(enabled=True),
        )
    )


def test_model_builds_one_lifecycle_per_extra_subsystem() -> None:
    model = make_model()
    assert model.subsystem_lifecycle is not None
    assert set(model.subsystem_lifecycle.engines) == set(ALL_MODALITIES)
    assert model.device_lifecycle_engine is not None


def test_flat_wrist_device_masks_core_vitals_but_not_an_owned_band() -> None:
    model = make_model()
    assert model.device_fleet is not None
    assert model.subsystem_lifecycle is not None
    position = model.device_fleet.kinds.index(THORACIC_EIT_ACOUSTIC_BAND)
    local_idx = int(np.flatnonzero(model.device_fleet.ownership[:, position])[0])
    agent = model.citizen_agents[local_idx]
    thoracic_engine = model.subsystem_lifecycle.engines[THORACIC_EIT_ACOUSTIC_BAND.name]
    thoracic_engine.status[local_idx] = DeviceStatus.ACTIVE

    core_columns = list(model.device_fleet.base_columns)
    thoracic_column = model.channel_set.index(THORACIC_CHANNEL)
    agent.device_status = DeviceStatus.DEPLETED
    band_seen = 0
    for _ in range(40):
        matrix = model._fleet_observed_matrix(2.0, 0.0)
        row = model._agent_observed_channels(agent, matrix)
        assert row is not None
        assert not row[core_columns].any()
        band_seen += int(row[thoracic_column])
    assert band_seen > 0


def test_an_unadopted_wearable_reports_nothing_from_any_subsystem() -> None:
    model = make_model()
    agent = model.citizen_agents[0]
    agent.device_status = DeviceStatus.NOT_ADOPTED
    row = model._agent_observed_channels(agent, model._fleet_observed_matrix(2.0, 0.0))
    assert row is not None
    assert not row.any()


def test_band_owners_keep_learning_while_the_watch_is_off() -> None:
    """A dead watch must not stall the baselines of a band that is still worn."""
    model = make_model(seed=17)
    assert model.device_fleet is not None
    position = model.device_fleet.kinds.index(THORACIC_EIT_ACOUSTIC_BAND)
    owners = np.flatnonzero(model.device_fleet.ownership[:, position])
    for local_idx in owners:
        model.citizen_agents[int(local_idx)].device_status = DeviceStatus.NOT_WORN
    before = np.array(
        [model.citizen_agents[int(i)].baseline.n_samples for i in owners], dtype=np.int64
    )
    for _ in range(12):
        model.step()
    after = np.array(
        [model.citizen_agents[int(i)].baseline.n_samples for i in owners], dtype=np.int64
    )
    assert np.any(after > before)


@pytest.mark.parametrize("backend", ["hex", "rect"])
def test_subsystem_lifecycle_runs_on_both_spatial_backends(backend: str) -> None:
    model = make_model(seed=23, backend=backend)
    for _ in range(20):
        model.step()
    assert model.subsystem_lifecycle is not None
    for engine in model.subsystem_lifecycle.engines.values():
        assert np.all(np.isfinite(engine.battery_levels))
        assert engine.battery_levels.min() >= 0.0
    for agent in model.citizen_agents:
        assert np.all(np.isfinite(agent.baseline.ema))


def test_lifecycle_metrics_expose_each_subsystem() -> None:
    model = make_model()
    model.step()
    metrics = model._device_lifecycle_metrics()
    assert "wearables_active" in metrics
    for name in ALL_MODALITIES:
        assert metrics[f"subsystem_{name}_active"] >= 0
        assert 0.0 <= float(metrics[f"subsystem_{name}_battery"]) <= 2.0


def test_lifecycle_without_devices_has_no_subsystem_state() -> None:
    model = GarlandModel(
        SimulationConfig(
            n_agents=200,
            n_steps=5,
            seed=9,
            device_lifecycle=DeviceLifecycleConfig(enabled=True),
        )
    )
    assert model.subsystem_lifecycle is None
    model.step()
    metrics = model._device_lifecycle_metrics()
    assert not any(key.startswith("subsystem_") for key in metrics)
