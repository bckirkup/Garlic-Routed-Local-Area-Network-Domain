from __future__ import annotations

import numpy as np
import pytest

from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.perturbations import PerturbationCause
from garland.simulation import GarlandModel, SimulationConfig


def _activity_count(rate: float) -> int:
    engine = ConfounderEngine(
        80,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=rate,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
        ),
        np.random.default_rng(42),
    )
    mask = np.ones(80, dtype=bool)
    return sum(
        len(step.affected_agents_by_cause.get(PerturbationCause.EXERCISE, set()))
        for step in (
            engine.step(step, 12.0, mask)
            for step in range(48)
        )
    )


def test_individual_confounder_rate_is_sensitive_and_bounded():
    counts = [_activity_count(rate) for rate in (0.0, 0.05, 0.2)]

    assert counts == sorted(counts)
    assert counts[-1] - counts[0] > 0


def test_sleep_disruption_rate_is_sensitive():
    counts = []
    for rate in (0.0, 0.1, 0.5):
        engine = ConfounderEngine(
            80,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=rate,
                sensor_artifact_probability=0.0,
                sleep_disruption_delay_steps=1,
                sleep_disruption_duration_steps=12,
            ),
            np.random.default_rng(42),
        )
        mask = np.ones(80, dtype=bool)
        counts.append(
            sum(
                len(
                    engine.step(step, 8.0, mask).affected_agents_by_cause.get(
                        PerturbationCause.SLEEP_DISRUPTION, set()
                    )
                )
                for step in range(24 * 12 + 1)
            )
        )

    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_sensor_artifact_probability_is_sensitive():
    counts = []
    for probability in (0.0, 0.25, 1.0):
        engine = ConfounderEngine(
            80,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=probability,
            ),
            np.random.default_rng(42),
        )
        mask = np.ones(80, dtype=bool)
        counts.append(
            sum(
                len(
                    engine.step(step, 8.0, mask, set(range(80)))
                    .affected_agents_by_cause.get(PerturbationCause.SENSOR_ARTIFACT, set())
                )
                for step in range(8)
            )
        )

    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_heat_wave_is_shared_and_has_instance_footprint():
    engine = ConfounderEngine(
        10,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            heat_wave_start_step=2,
            heat_wave_duration_steps=3,
        ),
        np.random.default_rng(42),
        zone_ids=(1, 2, 3),
    )
    mask = np.ones(10, dtype=bool)

    inactive = engine.step(1, 12.0, mask)
    active = engine.step(2, 12.0, mask)

    assert not inactive.heat_wave_active
    assert active.heat_wave_active
    assert active.heat_wave_instance_id == "heat_0"
    assert active.heat_wave_start_step == 2
    assert active.heat_wave_end_step == 5
    assert len(active.affected_agents_by_cause[PerturbationCause.HEAT_WAVE]) == 10
    assert all(
        contribution.cause == PerturbationCause.HEAT_WAVE
        for contributions in active.contributions.values()
        for contribution in contributions
    )


def test_confounders_are_hazard_independent():
    config = ConfoundersConfig(
        enabled=True,
        exercise_rate=0.1,
        sleep_disruption_rate=0.1,
        sensor_artifact_probability=0.5,
    )
    first = ConfounderEngine(20, config, np.random.default_rng(7))
    second = ConfounderEngine(20, config, np.random.default_rng(7))
    mask = np.ones(20, dtype=bool)
    for step in range(20):
        expected = first.step(step, 12.0, mask, {1, 2})
        actual = second.step(step, 12.0, mask, {1, 2})
        assert expected.affected_agents_by_cause == actual.affected_agents_by_cause


def test_disabled_confounders_have_zero_metrics_and_preserve_round_one():
    base = SimulationConfig(
        n_agents=80,
        wearable_fraction=0.5,
        n_steps=24,
        seed=11,
        mobility_model="static",
        world_settling_steps=0,
    )
    disabled = GarlandModel(base)
    disabled.run()

    zero_sources = SimulationConfig(
        n_agents=80,
        wearable_fraction=0.5,
        n_steps=24,
        seed=11,
        mobility_model="static",
        world_settling_steps=0,
        confounders=ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            heat_wave_duration_steps=0,
        ),
    )
    enabled_without_sources = GarlandModel(zero_sources)
    enabled_without_sources.run()
    disabled_summary = disabled.metrics.summary()
    zero_summary = enabled_without_sources.metrics.summary()

    for key in (
        "total_broadcasts",
        "total_responses",
        "total_epsilon",
        "background_rate",
    ):
        assert disabled_summary[key] == zero_summary[key]
    assert zero_summary["confounder_contributions_by_cause"] == {}
    assert zero_summary["heat_wave_active_steps"] == 0


@pytest.mark.parametrize("spatial_backend", ["hex", "rect"])
def test_enabled_confounders_report_activity_on_both_backends(spatial_backend: str):
    config = SimulationConfig(
        n_agents=100,
        wearable_fraction=0.5,
        n_steps=36,
        seed=3,
        mobility_model="static",
        spatial_backend=spatial_backend,
        world_settling_steps=0,
        confounders=ConfoundersConfig(
            enabled=True,
            exercise_rate=0.15,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            heat_wave_start_step=6,
            heat_wave_duration_steps=12,
        ),
    )
    model = GarlandModel(config)
    model.run()
    summary = model.metrics.summary()

    assert summary["confounder_contributions_by_cause"]["exercise"] > 0
    assert summary["confounder_contributions_by_cause"]["heat_wave"] > 0
    assert summary["heat_wave_active_steps"] == 12
    assert summary["heat_wave_instances"]["heat_0"]["zone_ids"]
