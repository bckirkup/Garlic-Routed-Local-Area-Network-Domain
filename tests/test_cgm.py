"""Regression coverage for the diabetic-gated CGM modality."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from garland.channels import CORE_VITALS, INTERSTITIAL_GLUCOSE, ChannelSet
from garland.config import config_from_dict, config_to_dict
from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.devices import CGM_PATCH, DeviceFleet, DeviceFleetConfig, build_channel_set
from garland.host_phenotypes import HostPhenotypeConfig, HostPhenotypes, host_presentation
from garland.modality_signatures import (
    contact_artifact_axes,
    exertion_axes,
    heat_strain_axes,
    infection_axes,
    irritant_axes,
    modality_delta,
)
from garland.perturbations import PerturbationCause
from garland.simulation import GarlandModel, SimulationConfig


def _meal_engine(
    channel_set: ChannelSet,
    seed: int = 42,
    jitter_steps: int = 0,
) -> ConfounderEngine:
    return ConfounderEngine(
        12,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            meal_excursion_jitter_steps=jitter_steps,
            meal_excursion_rise_steps=6,
            meal_excursion_decay_steps=24,
        ),
        np.random.default_rng(seed),
        channel_set=channel_set,
    )


def _meal_delta(step: object, channel_set: ChannelSet, agent: int = 0) -> float:
    contributions = step.contributions.get(agent, ())
    for contribution in contributions:
        if contribution.cause == PerturbationCause.MEAL_EXCURSION:
            return float(contribution.delta[channel_set.index(INTERSTITIAL_GLUCOSE.name)])
    return 0.0


def test_cgm_ownership_is_subset_of_diabetic_hosts() -> None:
    n = 2_000
    host_rng = np.random.default_rng(42)
    hosts = HostPhenotypes(
        n,
        HostPhenotypeConfig(enabled=True, diabetic_fraction=0.2),
        host_rng,
        np.full(n, 2, dtype=np.int8),
        ConfoundersConfig(),
        demographics_enabled=False,
    )
    fleet = DeviceFleet(
        n,
        DeviceFleetConfig(enabled=True, adoption={"cgm_patch": 0.5}),
        np.random.default_rng(17),
        eligibility_by_kind={"cgm_patch": hosts.diabetic},
    )

    position = next(i for i, kind in enumerate(fleet.kinds) if kind == CGM_PATCH)
    assert np.all(~fleet.ownership[:, position] | hosts.diabetic)


def test_cgm_without_hosts_warns_and_remains_ungated(caplog: pytest.LogCaptureFixture) -> None:
    config = SimulationConfig(
        n_agents=100,
        n_steps=1,
        wearable_fraction=0.5,
        devices=DeviceFleetConfig(enabled=True, adoption={"cgm_patch": 0.2}),
    )
    with caplog.at_level(logging.WARNING):
        model = GarlandModel(config)
    assert "cgm_patch adoption is ungated" in caplog.text
    cgm_position = next(i for i, kind in enumerate(model.device_fleet.kinds) if kind == CGM_PATCH)
    assert np.count_nonzero(model.device_fleet.ownership[:, cgm_position]) == 10


def test_cgm_modality_directionality() -> None:
    channel_set = build_channel_set((CGM_PATCH,))
    glucose = channel_set.index(INTERSTITIAL_GLUCOSE.name)
    infection = modality_delta(infection_axes(1.0), channel_set)[glucose]
    exertion = modality_delta(exertion_axes(1.0), channel_set)[glucose]
    irritant = modality_delta(irritant_axes(1.0), channel_set)[glucose]
    heat = modality_delta(heat_strain_axes(1.0), channel_set)[glucose]

    assert infection > 0.0
    assert exertion < 0.0
    assert irritant == pytest.approx(0.0)
    assert heat > 0.0
    assert heat < infection
    artifact = modality_delta(contact_artifact_axes(1.0), channel_set)[glucose]
    assert artifact == pytest.approx(0.0)


def test_diabetic_infection_has_larger_glucose_excursion() -> None:
    channel_set = build_channel_set((CGM_PATCH,))
    axes = infection_axes(1.0)
    normal = modality_delta(axes, channel_set)
    diabetic_axes, _ = host_presentation(axes, {}, True, False)
    diabetic = modality_delta(diabetic_axes, channel_set)
    glucose = channel_set.index(INTERSTITIAL_GLUCOSE.name)
    assert diabetic[glucose] > normal[glucose]


def test_meal_excursion_has_windowed_glucose_envelope() -> None:
    channel_set = build_channel_set((CGM_PATCH,))
    engine = _meal_engine(channel_set)
    mask = np.ones(12, dtype=bool)

    steps = {step: engine.step(step, 7.5, mask) for step in range(89, 121)}
    before = steps[89]
    onset = steps[90]
    rising = steps[93]
    outside = steps[120]

    assert _meal_delta(before, channel_set) == pytest.approx(0.0)
    assert _meal_delta(onset, channel_set) > 0.0
    assert _meal_delta(rising, channel_set) > _meal_delta(onset, channel_set)
    assert _meal_delta(outside, channel_set) == pytest.approx(0.0)


def test_meal_excursion_is_absent_without_glucose_channel() -> None:
    engine = _meal_engine(CORE_VITALS)
    step = engine.step(90, 7.5, np.ones(12, dtype=bool))
    assert step.contributions == {}


def test_meal_excursion_draws_are_seed_deterministic() -> None:
    channel_set = build_channel_set((CGM_PATCH,))
    first = _meal_engine(channel_set, seed=9)
    second = _meal_engine(channel_set, seed=9)
    mask = np.ones(12, dtype=bool)

    for step in range(90, 98):
        left = first.step(step, 7.5, mask)
        right = second.step(step, 7.5, mask)
        assert _meal_delta(left, channel_set) == pytest.approx(_meal_delta(right, channel_set))


def test_meal_excursions_are_not_fleet_wide_identical() -> None:
    channel_set = build_channel_set((CGM_PATCH,))
    engine = _meal_engine(channel_set, seed=9, jitter_steps=6)
    mask = np.ones(12, dtype=bool)
    observed: dict[int, list[tuple[int, float]]] = {0: [], 1: []}

    for step in range(84, 121):
        result = engine.step(step, 7.5, mask)
        for agent in observed:
            value = _meal_delta(result, channel_set, agent)
            if value > 0.0:
                observed[agent].append((step, value))

    assert observed[0]
    assert observed[1]
    assert observed[0][0][0] != observed[1][0][0] or not np.isclose(
        observed[0][0][1], observed[1][0][1]
    )


def test_meal_excursion_jitter_round_trips_in_config() -> None:
    config = SimulationConfig(
        confounders=ConfoundersConfig(meal_excursion_jitter_steps=11),
    )
    restored = config_from_dict(config_to_dict(config))
    assert restored.confounders.meal_excursion_jitter_steps == 11


def test_cgm_hard_floor_holds_after_large_negative_delta() -> None:
    channel_set = ChannelSet((INTERSTITIAL_GLUCOSE,))
    assert channel_set.clamp(np.array([-1_000.0]))[0] == pytest.approx(40.0)


def test_zero_cgm_adoption_does_not_widen_fleet() -> None:
    no_cgm = DeviceFleet(
        100,
        DeviceFleetConfig(enabled=True),
        np.random.default_rng(1),
    )
    zero_cgm = DeviceFleet(
        100,
        DeviceFleetConfig(enabled=True, adoption={"cgm_patch": 0.0}),
        np.random.default_rng(1),
    )
    position = next(i for i, kind in enumerate(zero_cgm.kinds) if kind == CGM_PATCH)
    assert np.count_nonzero(zero_cgm.ownership[:, position]) == 0
    assert INTERSTITIAL_GLUCOSE.name not in no_cgm.channel_set.names
