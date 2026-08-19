from __future__ import annotations

import numpy as np
import pytest

from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.hazards import SEIRConfig
from garland.perturbations import PerturbationCause
from garland.privacy import PrivacyConfig
from garland.simulation import GarlandModel, SimulationConfig


def _fire_engine(**overrides: object) -> ConfounderEngine:
    values: dict[str, object] = {
        "enabled": True,
        "exercise_rate": 0.0,
        "sleep_disruption_rate": 0.0,
        "sensor_artifact_probability": 0.0,
        "block_fire_start_step": 0,
        "block_fire_duration_steps": 6,
        "block_fire_center_x": 50.0,
        "block_fire_center_y": 0.0,
        "block_fire_radius_m": 20.0,
        "block_fire_materiality_floor": 0.25,
        "block_fire_amplitude_jitter": 0.0,
    }
    values.update(overrides)
    return ConfounderEngine(
        101,
        ConfoundersConfig(**values),
        np.random.default_rng(42),
        agent_x=np.linspace(0.0, 100.0, 101),
        agent_y=np.zeros(101),
        exposure_rng=np.random.default_rng(43),
    )


def _victory_engine(**overrides: object) -> ConfounderEngine:
    values: dict[str, object] = {
        "enabled": True,
        "exercise_rate": 0.0,
        "sleep_disruption_rate": 0.0,
        "sensor_artifact_probability": 0.0,
        "victory_start_step": 0,
        "victory_duration_steps": 8,
        "victory_onset_jitter_steps": 3,
        "victory_participation_fraction": 1.0,
        "victory_amplitude_jitter": 0.0,
    }
    values.update(overrides)
    return ConfounderEngine(
        200,
        ConfoundersConfig(**values),
        np.random.default_rng(42),
        exposure_rng=np.random.default_rng(43),
    )


def _fire_instance(engine: ConfounderEngine, step: int = 0):
    return engine.step(
        step,
        15.0,
        np.ones(engine.n_agents, dtype=bool),
        agent_x=engine.initial_agent_x,
        agent_y=engine.initial_agent_y,
    ).benign_instances["block_fire_0"]


def test_fire_radius_and_elderly_sensitivity_have_real_margins():
    counts = [
        len(
            _fire_instance(
                _fire_engine(block_fire_radius_m=radius),
            ).current_agents
        )
        for radius in (10.0, 20.0, 35.0)
    ]
    assert counts == sorted(counts)
    assert counts[-1] - counts[0] > 10

    totals = []
    for elderly_weight in (0.0, 0.5, 1.0):
        engine = _fire_engine(block_fire_elderly_weight=elderly_weight)
        engine.elderly[:] = True
        result = engine.step(0, 15.0, np.ones(101, dtype=bool))
        totals.append(
            sum(
                float(contribution.delta[0])
                for contributions in result.contributions.values()
                for contribution in contributions
                if contribution.cause is PerturbationCause.IRRITANT_EXPOSURE
            )
        )
    assert totals == sorted(totals)
    assert totals[-1] - totals[0] > 10.0


def test_fire_distance_signature_bounds_and_negative_control():
    engine = _fire_engine(block_fire_radius_m=30.0)
    engine.elderly[:] = False
    distance_weights = engine._block_fire_distance_weights(
        engine.initial_agent_x,
        engine.initial_agent_y,
    )
    assert np.all((0.0 <= distance_weights) & (distance_weights <= 1.0))
    result = engine.step(0, 15.0, np.ones(101, dtype=bool))
    respiratory = {
        idx: contribution.delta[2]
        for idx, contributions in result.contributions.items()
        for contribution in contributions
        if contribution.cause is PerturbationCause.IRRITANT_EXPOSURE
    }
    assert respiratory[50] > respiratory[60] > respiratory[80]
    assert all(np.isfinite(value) for value in respiratory.values())
    assert all(value > 0.0 for value in respiratory.values())
    fire_deltas = [
        contribution.delta[3]
        for contributions in result.contributions.values()
        for contribution in contributions
        if contribution.cause is PerturbationCause.IRRITANT_EXPOSURE
    ]
    assert all(value == pytest.approx(0.0) for value in fire_deltas)
    assert set(result.contributions) <= set(np.flatnonzero(np.ones(101, dtype=bool)))

    base = _fire_instance(_fire_engine(sensor_artifact_probability=0.0))
    unrelated = _fire_instance(_fire_engine(sensor_artifact_probability=1.0))
    assert base.current_agents == unrelated.current_agents


def test_new_events_are_disabled_by_default():
    engine = ConfounderEngine(
        20,
        ConfoundersConfig(enabled=True, exercise_rate=0.0, sleep_disruption_rate=0.0),
        np.random.default_rng(42),
    )
    result = engine.step(0, 15.0, np.ones(20, dtype=bool))
    assert result.contributions == {}
    assert result.benign_instances == {}


def test_fire_membership_uses_current_positions():
    engine = _fire_engine(block_fire_center_x=0.0, block_fire_radius_m=10.0)
    mask = np.ones(101, dtype=bool)
    first = engine.step(
        0,
        15.0,
        mask,
        agent_x=np.zeros(101),
        agent_y=np.zeros(101),
    )
    assert 0 in first.benign_instances["block_fire_0"].current_agents
    second = engine.step(
        1,
        15.0,
        mask,
        agent_x=np.full(101, 100.0),
        agent_y=np.zeros(101),
    )
    assert 0 not in second.benign_instances["block_fire_0"].current_agents


def test_block_fire_locality_scales_across_spatial_backends():
    for backend in ("hex", "rect"):
        base = dict(
            n_agents=1000,
            wearable_fraction=1.0,
            n_steps=1,
            seed=91,
            mobility_model="static",
            spatial_backend=backend,
            world_settling_steps=0,
            seir=SEIRConfig(initial_infected=0),
            plumes=[],
        )
        probe = GarlandModel(
            SimulationConfig(
                **base,
                confounders=ConfoundersConfig(enabled=False),
            )
        )
        center_x = float(probe.agent_x[0])
        center_y = float(probe.agent_y[0])
        occupied_cell_count = len(set(probe.agent_cell_ids))
        spans = []
        for radius in (200.0, 1000.0):
            model = GarlandModel(
                SimulationConfig(
                    **base,
                    confounders=ConfoundersConfig(
                        enabled=True,
                        exercise_rate=0.0,
                        sleep_disruption_rate=0.0,
                        sensor_artifact_probability=0.0,
                        block_fire_duration_steps=1,
                        block_fire_center_x=center_x,
                        block_fire_center_y=center_y,
                        block_fire_radius_m=radius,
                        block_fire_materiality_floor=0.25,
                        block_fire_amplitude_jitter=0.0,
                    ),
                )
            )
            model.step()
            instance = model._confounder_step.benign_instances["block_fire_0"]
            spans.append(len({int(model.agent_cell_ids[idx]) for idx in instance.current_agents}))
        assert spans[1] > spans[0] + 3
        assert spans[0] < 0.5 * occupied_cell_count


def test_victory_fan_and_participation_sensitivity():
    fan_counts = []
    participation_counts = []
    mask = np.ones(200, dtype=bool)
    for fraction in (0.1, 0.25, 0.5):
        engine = _victory_engine(victory_fan_fraction=fraction)
        result = engine.step(3, 3.0, mask)
        fan_counts.append(len(result.benign_instances["victory_0"].current_agents))
    for fraction in (0.0, 0.5, 1.0):
        engine = _victory_engine(victory_participation_fraction=fraction)
        result = engine.step(3, 3.0, mask)
        participation_counts.append(len(result.benign_instances["victory_0"].current_agents))
    assert fan_counts == sorted(fan_counts)
    assert fan_counts[-1] - fan_counts[0] > 10
    assert participation_counts == sorted(participation_counts)
    assert participation_counts[-1] - participation_counts[0] > 10


def test_victory_is_synchronized_and_decays_after_peak():
    engine = _victory_engine(victory_fan_fraction=1.0)
    mask = np.ones(200, dtype=bool)
    engine.step(0, 3.0, mask)
    participating = engine.victory_participating
    victory_spread = int(np.ptp(engine.victory_onset_steps[participating]))

    individual = _victory_engine(
        victory_duration_steps=0,
        victory_fan_fraction=0.0,
        sleep_disruption_rate=1.0,
        sleep_disruption_delay_steps=96,
        sleep_disruption_delay_jitter_steps=24,
    )
    individual.step(22 * 12, 22.0, mask)
    individual_spread = int(np.ptp(22 * 12 + individual.sleep_delay))
    assert victory_spread <= 3
    assert individual_spread > victory_spread

    selected = int(np.flatnonzero(participating)[0])
    onset = int(engine.victory_onset_steps[selected])
    values = []
    for step in range(onset, onset + 8):
        result = engine.step(step, 3.0, mask)
        values.append(
            float(
                next(
                    contribution.delta[0]
                    for contribution in result.contributions[selected]
                    if contribution.cause is PerturbationCause.SLEEP_DISRUPTION
                )
            )
        )
    assert values == sorted(values, reverse=True)


def _warrant_model(backend: str, confounders: ConfoundersConfig) -> GarlandModel:
    """A day-long static run whose only perturbation is `confounders`."""
    return GarlandModel(
        SimulationConfig(
            n_agents=100,
            wearable_fraction=0.8,
            n_steps=24,
            seed=27,
            mobility_model="static",
            spatial_backend=backend,
            world_settling_steps=0,
            seir=SEIRConfig(initial_infected=0),
            plumes=[],
            # These warrant assertions use the calibrated operating
            # point; respondent-basis dilation is exercised separately.
            privacy=PrivacyConfig(dilation_basis="residents"),
            confounders=confounders,
        )
    )


def _assert_detections_partition(summary):
    """Every detection lands in exactly one warrant class."""
    assert summary["total_detection_events"] == sum(
        summary[key]
        for key in (
            "target_detections",
            "actionable_non_target_detections",
            "explained_detections",
            "artifact_detections",
            "unexplained_detections",
        )
    )


def test_victory_membership_and_model_warrants_on_both_backends():
    for backend in ("hex", "rect"):
        fire = _warrant_model(
            backend,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
                block_fire_duration_steps=24,
                block_fire_center_x=5000.0,
                block_fire_center_y=5000.0,
                block_fire_radius_m=10000.0,
                block_fire_materiality_floor=0.1,
                block_fire_hr_delta=12.0,
                block_fire_hrv_delta=-8.0,
                block_fire_respiratory_delta=8.0,
            ),
        )
        fire.run()
        fire_summary = fire.metrics.summary()
        assert fire_summary["actionable_non_target_detections"] > 0
        assert fire_summary["target_detections"] == 0
        _assert_detections_partition(fire_summary)

        victory = _warrant_model(
            backend,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
                victory_duration_steps=24,
                victory_fan_fraction=1.0,
                victory_participation_fraction=1.0,
                victory_hr_delta=12.0,
                victory_hrv_delta=-8.0,
                victory_temperature_delta=0.1,
            ),
        )
        wearable = set(np.flatnonzero(victory.has_wearable))
        victory.step()
        instance = victory._confounder_step.benign_instances["victory_0"]
        assert instance.current_agents <= wearable
        victory.run()
        summary = victory.metrics.summary()
        assert summary["explained_detections"] > 0
        assert summary["actionable_non_target_detections"] == 0
        _assert_detections_partition(summary)
