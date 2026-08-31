"""Regression coverage for host-dependent presentation and device needs."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.demographics import AGE_BANDS, DemographicsConfig
from garland.devices import HEARABLE, DeviceFleet, DeviceFleetConfig
from garland.hazards import SEIRConfig, SEIREngine, SEIRState
from garland.host_phenotypes import (
    HostPhenotypeConfig,
    HostPhenotypes,
    host_presentation,
)
from garland.modality_signatures import infection_axes
from garland.simulation import GarlandModel, SimulationConfig


def _ages(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int8)


def test_disabled_hosts_preserve_seeded_summary() -> None:
    base = SimulationConfig(
        n_agents=120,
        wearable_fraction=0.2,
        grid_width=1000.0,
        grid_height=1000.0,
        n_steps=4,
        seed=19,
    )
    explicit = SimulationConfig(**{**base.__dict__, "hosts": HostPhenotypeConfig(enabled=False)})
    assert GarlandModel(base).run().summary() == GarlandModel(explicit).run().summary()


def test_host_sampling_is_seeded_and_susceptibility_is_bounded() -> None:
    config = HostPhenotypeConfig(
        enabled=True,
        diabetic_fraction=0.4,
        law_enforcement_fraction=0.2,
        assistive_need_fraction=0.2,
    )
    ages = _ages([AGE_BANDS.index(name) for name in ("adult", "older_adult", "elderly") * 40])
    first = HostPhenotypes(
        len(ages), config, np.random.default_rng(7), ages, ConfoundersConfig(), True
    )
    second = HostPhenotypes(
        len(ages), config, np.random.default_rng(7), ages, ConfoundersConfig(), True
    )
    assert np.array_equal(first.diabetic, second.diabetic)
    assert np.all(np.isfinite(first.susceptibility_multiplier()))
    assert np.all((first.susceptibility_multiplier() >= 1.0))
    assert np.all(first.susceptibility_multiplier() <= 3.0)
    assert np.all(first.hearable_eligible[first.frail_elderly])


def test_diabetic_and_frail_presentation_changes_are_directional() -> None:
    axes = infection_axes(0.5)
    core = {
        "body_temperature": 1.5,
        "heart_rate": 15.0,
        "hrv_rmssd": -15.0,
        "respiratory_rate": 5.0,
    }
    diabetic_axes, diabetic_core = host_presentation(axes, core, True, False)
    frail_axes, frail_core = host_presentation(axes, core, False, True)
    assert diabetic_core["body_temperature"] < core["body_temperature"]
    assert diabetic_axes.hypovolemia > axes.hypovolemia
    assert frail_core["body_temperature"] == pytest.approx(core["body_temperature"] * 0.5)
    assert frail_axes.activity_withdrawal > axes.activity_withdrawal
    assert frail_axes.pulmonary_involvement > axes.pulmonary_involvement


def test_diabetic_course_stretches_and_frail_temperature_is_blunted() -> None:
    engine = SEIREngine(SEIRConfig(initial_infected=0))
    engine.states = np.asarray([SEIRState.INFECTIOUS], dtype=np.int8)
    engine.infection_step = np.asarray([0], dtype=np.int32)
    engine.host_diabetic = np.asarray([True])
    engine.host_frail_elderly = np.asarray([False])
    diabetic = engine.biometric_perturbation(0, 576)
    engine.host_diabetic = np.asarray([False])
    normal = engine.biometric_perturbation(0, 576)
    assert diabetic[3] < normal[3]
    engine.host_diabetic = np.asarray([False])
    engine.host_frail_elderly = np.asarray([True])
    frail = engine.biometric_perturbation(0, 576)
    assert frail[3] == pytest.approx(normal[3] * 0.5)


def test_host_susceptibility_increases_controlled_attack_rate() -> None:
    n_targets = 1000
    engine = SEIREngine(SEIRConfig(initial_infected=0, beta=0.2))
    engine.initialize(
        n_targets + 1,
        np.random.default_rng(1),
        np.zeros(n_targets + 1, dtype=np.float32),
        np.zeros(n_targets + 1, dtype=np.float32),
    )
    engine.outbreak_origin[0] = "controlled"
    targets = np.arange(1, n_targets + 1, dtype=np.intp)
    susceptible = np.ones(n_targets + 1, dtype=np.float64)
    susceptible[1 : n_targets // 2 + 1] = 2.24
    positions = np.zeros(n_targets + 1, dtype=np.float32)
    outcomes = np.zeros((2, 200), dtype=np.float64)
    for trial in range(outcomes.shape[1]):
        exposed = engine._proximity_transmissions(
            np.asarray([0], dtype=np.intp),
            targets,
            positions,
            positions,
            np.random.default_rng(10 + trial),
            susceptible,
        )
        selected = np.asarray(list(exposed), dtype=np.intp)
        outcomes[0, trial] = np.count_nonzero(selected < n_targets // 2 + 1)
        outcomes[1, trial] = np.count_nonzero(selected >= n_targets // 2 + 1)
    assert outcomes[0].mean() > outcomes[1].mean()


def test_hearable_ownership_is_eligible_and_caps_with_warning(caplog) -> None:
    n = 40
    ages = _ages([AGE_BANDS.index("adult")] * n)
    eligible = np.zeros(n, dtype=bool)
    eligible[:3] = True
    with caplog.at_level(logging.WARNING):
        fleet = DeviceFleet(
            n,
            DeviceFleetConfig(enabled=True, adoption={HEARABLE.name: 0.8}),
            np.random.default_rng(3),
            age_bands=ages,
            demographics=DemographicsConfig(enabled=True),
            eligibility_by_kind={HEARABLE.name: eligible},
        )
    position = fleet.kinds.index(HEARABLE)
    assert np.all(~fleet.ownership[:, position] | eligible)
    assert int(np.count_nonzero(fleet.ownership[:, position])) == 3
    assert "capped by eligibility" in caplog.text


def test_law_enforcement_sleep_disruption_rate_is_elevated() -> None:
    n = 1000
    law = np.zeros(n, dtype=bool)
    law[:500] = True
    config = ConfoundersConfig(
        enabled=True,
        sleep_disruption_rate=0.1,
        sleep_disruption_delay_steps=1,
        sleep_disruption_delay_jitter_steps=0,
        sleep_disruption_duration_steps=2,
        exercise_rate=0.0,
    )
    engine = ConfounderEngine(
        n,
        config,
        np.random.default_rng(11),
        host_law_enforcement=law,
    )
    engine.step(22 * 12, 23.0, np.ones(n, dtype=bool))
    assert np.count_nonzero(engine.sleep_remaining[:500]) > np.count_nonzero(
        engine.sleep_remaining[500:]
    )
