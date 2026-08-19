"""Shared fixtures for the per-device modality suites.

Every modality suite asks the same structural questions of its device — does an
adopter's vector widen, does a non-adopter's stay structurally missing, does the
subsystem run on its own cell, do the core vitals survive the widening — and
differs only in the channels and calibration it asserts on. These helpers hold
the structural half so each suite carries only its own physiology.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import CORE_VITALS, ChannelSet
from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.device_lifecycle import DeviceLifecycleConfig
from garland.devices import DeviceFleetConfig, DeviceKind
from garland.hazards import SEIRConfig, SEIREngine, SEIRState
from garland.perturbations import PerturbationCause
from garland.simulation import GarlandModel, SimulationConfig

CONFOUNDER_AGENTS = 16


def channel_value(delta: np.ndarray, channel_set: ChannelSet, name: str) -> float:
    """Read one named channel out of a channel-set-shaped delta vector."""
    return float(delta[channel_set.index(name)])


def infectious_engine(seed: int = 3, n_agents: int = 8) -> SEIREngine:
    """An engine whose agent 0 is infectious, for signature probing."""
    engine = SEIREngine(SEIRConfig(initial_infected=1))
    engine.initialize(n_agents, np.random.default_rng(seed))
    engine.states[0] = SEIRState.INFECTIOUS
    return engine


def modality_model(
    device: DeviceKind,
    backend: str = "rect",
    adoption: float = 1.0,
    seed: int = 31,
    lifecycle: bool = False,
) -> GarlandModel:
    """A small model in which an `adoption` fraction of wearers own `device`."""
    return GarlandModel(
        SimulationConfig(
            n_agents=200,
            n_steps=20,
            wearable_fraction=0.5,
            seed=seed,
            spatial_backend=backend,
            device_lifecycle=DeviceLifecycleConfig(enabled=lifecycle),
            devices=DeviceFleetConfig(
                enabled=True,
                adoption={device.name: adoption},
            ),
        )
    )


def step_model(model: GarlandModel, steps: int) -> GarlandModel:
    for _ in range(steps):
        model.step()
    return model


def assert_model_runs(model: GarlandModel, present: str, steps: int = 18) -> None:
    """The model advances with the device adopted and keeps finite baselines."""
    assert model.device_fleet is not None
    assert model.channel_set.has(present)
    step_model(model, steps)
    assert model.citizen_agents
    for agent in model.citizen_agents:
        assert np.all(np.isfinite(agent.baseline.ema))


def assert_channels_structurally_missing(
    model: GarlandModel,
    device: DeviceKind,
    names: tuple[str, ...],
    hour_of_day: float = 12.0,
    activity: float = 0.5,
    seed: int = 4,
) -> None:
    """Nobody owns `device`, so its channels are never observed by anyone."""
    assert model.device_fleet is not None
    step_model(model, 6)
    assert model.device_fleet.owner_counts()[device.name] == 0
    mask = model.device_fleet.observed_matrix(hour_of_day, activity, np.random.default_rng(seed))
    for name in names:
        assert not mask[:, model.channel_set.index(name)].any()


def assert_subsystem_battery_is_independent(model: GarlandModel, device: DeviceKind) -> None:
    """The subsystem drains its own cell rather than tracking the watch's."""
    lifecycle = model.subsystem_lifecycle
    assert lifecycle is not None
    engine = lifecycle.engines[device.name]
    step_model(model, 12)
    assert np.all(np.isfinite(engine.battery_levels))
    assert np.all(engine.battery_levels >= 0.0)
    watch_batteries = np.array(
        [agent.battery_level for agent in model.citizen_agents if agent.has_wearable]
    )
    assert not np.allclose(engine.battery_levels[: watch_batteries.size], watch_batteries)


def assert_core_vitals_unchanged(channel_set: ChannelSet, steps_since: int = 400) -> None:
    """Widening the vector must not move the four channels everyone reports."""
    core = infectious_engine().biometric_perturbation(0, steps_since, CORE_VITALS)
    wide = infectious_engine().biometric_perturbation(0, steps_since, channel_set)
    for position, name in enumerate(CORE_VITALS.names):
        assert wide[channel_set.index(name)] == pytest.approx(core[position])


def confounder_deltas_by_cause(
    channel_set: ChannelSet,
    seed: int = 19,
    steps: int = 4,
) -> dict[PerturbationCause, np.ndarray]:
    """One delta vector per benign cause the confounder engine can fire.

    Artifact fires on posture/venue transitions, so every agent has to be
    transitioning for that arm to be exercised at all.
    """
    engine = ConfounderEngine(
        CONFOUNDER_AGENTS,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=1.0,
            sensor_artifact_probability=1.0,
        ),
        np.random.default_rng(seed),
        channel_set=channel_set,
    )
    by_cause: dict[PerturbationCause, np.ndarray] = {}
    for step_index in range(steps):
        step = engine.step(
            current_step=12 * 12 + step_index,
            hour_of_day=12.0,
            wearable_mask=np.ones(CONFOUNDER_AGENTS, dtype=bool),
            transition_indices=set(range(CONFOUNDER_AGENTS)),
        )
        for contributions in step.contributions.values():
            for contribution in contributions:
                by_cause.setdefault(contribution.cause, contribution.delta)
    return by_cause
