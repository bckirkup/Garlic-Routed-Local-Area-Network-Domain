from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from garland.adoption import AdoptionConfig
from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.hazards import PlumeConfig, SEIRConfig
from garland.perturbations import PerturbationCause
from garland.privacy import PrivacyConfig
from garland.simulation import GarlandModel, SimulationConfig
from garland.venues import VenueConfig, VenueEngine, VenueSystemConfig, VenueType


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
        for step in (engine.step(step, 12.0, mask) for step in range(48))
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
                    engine.step(step, 8.0, mask, set(range(80))).affected_agents_by_cause.get(
                        PerturbationCause.SENSOR_ARTIFACT, set()
                    )
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
            has_air_conditioning_fraction=0.0,
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
    heat_instance = active.benign_instances["heat_0"]
    assert not heat_instance.global_scope
    assert heat_instance.current_agents == set(range(10))
    assert all(
        contribution.cause == PerturbationCause.HEAT_WAVE
        for contributions in active.contributions.values()
        for contribution in contributions
    )
    amplitudes = {
        tuple(contribution.delta)
        for contributions in active.contributions.values()
        for contribution in contributions
    }
    assert len(amplitudes) > 1


def _heat_engine(**kwargs: object) -> ConfounderEngine:
    config = {
        "enabled": True,
        "exercise_rate": 0.0,
        "sleep_disruption_rate": 0.0,
        "sensor_artifact_probability": 0.0,
        "heat_wave_start_step": 0,
        "heat_wave_duration_steps": 2,
    }
    config.update(kwargs)
    return ConfounderEngine(
        100,
        ConfoundersConfig(**config),
        np.random.default_rng(42),
        agent_x=np.linspace(0.0, 100.0, 100),
        agent_y=np.zeros(100),
    )


def test_heat_exposure_knobs_grade_and_unrelated_source_is_a_negative_control():
    mask = np.ones(100, dtype=bool)
    mean_weights = []
    for fraction in (0.0, 0.5, 1.0):
        engine = _heat_engine(has_air_conditioning_fraction=fraction)
        mean_weights.append(float(np.mean(engine._heat_wave_weights(15.0, mask)[0])))
    assert mean_weights == sorted(mean_weights, reverse=True)
    assert mean_weights[0] - mean_weights[-1] > 0.25

    base = _heat_engine()
    unrelated = _heat_engine(sensor_artifact_probability=1.0)
    base_step = base.step(0, 15.0, mask, set(range(100)))
    unrelated_step = unrelated.step(0, 15.0, mask, set(range(100)))
    assert len(base_step.affected_agents_by_cause.get(PerturbationCause.HEAT_WAVE, set())) == len(
        unrelated_step.affected_agents_by_cause.get(PerturbationCause.HEAT_WAVE, set())
    )


def test_heat_vulnerability_and_island_gain_have_ordered_effects():
    mask = np.ones(100, dtype=bool)

    def mean_weight(**kwargs: object) -> float:
        engine = _heat_engine(
            has_air_conditioning_fraction=0.0,
            **kwargs,
        )
        return float(np.mean(engine._heat_wave_weights(15.0, mask)[0]))

    elderly_weights = [mean_weight(elderly_fraction=fraction) for fraction in (0.0, 0.5, 1.0)]
    worker_weights = [mean_weight(outdoor_worker_fraction=fraction) for fraction in (0.0, 0.5, 1.0)]
    assert elderly_weights == sorted(elderly_weights)
    assert worker_weights == sorted(worker_weights)
    assert elderly_weights[-1] > elderly_weights[0]
    assert worker_weights[-1] > worker_weights[0]

    low_gain = _heat_engine(
        has_air_conditioning_fraction=0.0,
        heat_island_gain=0.0,
    )
    high_gain = _heat_engine(
        has_air_conditioning_fraction=0.0,
        heat_island_gain=1.0,
    )
    low_weights = low_gain._heat_wave_weights(15.0, mask)[0]
    high_weights = high_gain._heat_wave_weights(15.0, mask)[0]
    assert high_weights[50] > low_weights[50]
    assert high_weights[-1] == pytest.approx(low_weights[-1])


def test_heat_diurnal_profile_and_night_air_conditioning_boundary():
    mask = np.ones(100, dtype=bool)
    engine = _heat_engine(has_air_conditioning_fraction=0.5)
    night_weights = engine._heat_wave_weights(3.0, mask)[0]
    afternoon_weights = engine._heat_wave_weights(15.0, mask)[0]
    assert float(np.mean(night_weights)) < float(np.mean(afternoon_weights))
    uncooled = ~engine.has_air_conditioning
    assert np.all(night_weights[uncooled] >= engine.config.heat_wave_night_floor)
    assert np.all(night_weights[engine.has_air_conditioning] > 0.0)


def test_heat_materiality_floor_separates_instance_footprint_from_perturbations():
    engine = _heat_engine(has_air_conditioning_fraction=0.5)
    mask = np.ones(100, dtype=bool)
    for step, hour in ((0, 15.0), (1, 3.0)):
        result = engine.step(step, hour, mask)
        perturbed = {
            idx
            for idx, contributions in result.contributions.items()
            if any(
                contribution.cause is PerturbationCause.HEAT_WAVE for contribution in contributions
            )
        }
        affected = result.benign_instances["heat_0"].current_agents
        assert affected < set(range(100))
        assert perturbed == set(range(100))


def test_heat_materiality_floor_grades_affected_set_size():
    mask = np.ones(100, dtype=bool)
    affected_counts = []
    for floor in (0.5, 1.5, 2.0):
        engine = _heat_engine(
            has_air_conditioning_fraction=0.0,
            heat_wave_materiality_floor=floor,
        )
        result = engine.step(0, 15.0, mask)
        affected_counts.append(len(result.benign_instances["heat_0"].current_agents))
    assert affected_counts == sorted(affected_counts, reverse=True)
    assert affected_counts[0] - affected_counts[-1] > 10


def test_heat_night_material_footprint_is_nonempty_and_uncooled():
    engine = _heat_engine()
    result = engine.step(0, 3.0, np.ones(100, dtype=bool))
    affected = result.benign_instances["heat_0"].current_agents
    assert affected
    assert all(not engine.has_air_conditioning[idx] for idx in affected)


def test_sleep_disruption_delay_jitter_breaks_synchronization():
    def onset_steps(jitter: int) -> set[int]:
        engine = ConfounderEngine(
            100,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=1.0,
                sleep_disruption_delay_steps=4,
                sleep_disruption_delay_jitter_steps=jitter,
                sleep_disruption_duration_steps=1,
                sensor_artifact_probability=0.0,
            ),
            np.random.default_rng(42),
        )
        mask = np.ones(100, dtype=bool)
        return {
            step
            for step in range(264, 300)
            if engine.step(step, 22.0 if step == 264 else 8.0, mask).affected_agents_by_cause.get(
                PerturbationCause.SLEEP_DISRUPTION
            )
        }

    assert len(onset_steps(3)) > 1
    assert len(onset_steps(0)) == 1


def test_venue_crowding_tracks_dynamic_membership_and_occupancy():
    venue = SimpleNamespace(
        venue_id="gym",
        venue_type="sporting_event",
        capacity=4,
    )
    membership = np.array([0, 0, 0, 0, -1, -1], dtype=np.int32)
    venue_engine = SimpleNamespace(
        venues=[venue],
        agents_at_venue=lambda index: np.flatnonzero(membership == index),
    )
    engine = ConfounderEngine(
        6,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            venue_crowding_rate=1.0,
            venue_crowding_duration_steps=3,
            venue_crowding_venue_types=("sporting_event",),
        ),
        np.random.default_rng(42),
        venue_engine=venue_engine,
    )
    mask = np.ones(6, dtype=bool)
    active = engine.step(0, 12.0, mask)
    assert active.benign_instances["venue_gym_0"].current_agents == {0, 1, 2, 3}
    membership[3] = -1
    membership[4] = 0
    moved = engine.step(1, 12.0, mask)
    assert moved.benign_instances["venue_gym_0"].current_agents == {0, 1, 2, 4}
    assert len(moved.affected_agents_by_cause[PerturbationCause.VENUE_CROWDING]) == 4


def test_venue_crowding_intensity_is_ordered_by_occupancy():
    venue = SimpleNamespace(
        venue_id="gym",
        venue_type=VenueType.SPORTING.value,
        capacity=10,
    )
    membership = np.full(12, -1, dtype=np.int32)
    venue_engine = SimpleNamespace(
        venues=[venue],
        agents_at_venue=lambda index: np.flatnonzero(membership == index),
    )
    magnitudes = []
    for occupancy in (2, 5, 10):
        membership[:] = -1
        membership[:occupancy] = 0
        engine = ConfounderEngine(
            12,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
                venue_crowding_rate=1.0,
                venue_crowding_venue_types=(VenueType.SPORTING.value,),
                venue_crowding_amplitude_jitter=0.0,
            ),
            np.random.default_rng(42),
            venue_engine=venue_engine,
        )
        step = engine.step(0, 12.0, np.ones(12, dtype=bool))
        magnitudes.append(float(np.linalg.norm(step.contributions[0][0].delta)))
    assert magnitudes == sorted(magnitudes)
    assert len(set(magnitudes)) == 3


def test_venue_crowding_uses_real_venue_engine_membership():
    venue_config = VenueConfig(
        venue_id="gathering",
        venue_type=VenueType.GATHERING.value,
        center_x=0.0,
        center_y=0.0,
        capacity=6,
    )
    venue_engine = VenueEngine(VenueSystemConfig(enabled=True, venues=[venue_config]))
    venue_engine.initialize(
        6,
        np.random.default_rng(7),
        np.zeros(6, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.zeros(6, dtype=np.int64),
    )
    venue_engine.current_venue_idx[:] = np.array([0, 0, 0, -1, -1, -1])
    engine = ConfounderEngine(
        6,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            venue_crowding_rate=1.0,
            venue_crowding_venue_types=(VenueType.GATHERING,),
        ),
        np.random.default_rng(42),
        venue_engine=venue_engine,
    )
    step = engine.step(0, 12.0, np.ones(6, dtype=bool))
    assert step.benign_instances["venue_gathering_0"].current_agents == {0, 1, 2}
    assert len(step.contributions) == 3


def test_benign_registry_prunes_expired_instances_and_amplitudes():
    engine = ConfounderEngine(
        8,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            heat_wave_start_step=0,
            heat_wave_duration_steps=1,
        ),
        np.random.default_rng(42),
    )
    mask = np.ones(8, dtype=bool)
    assert "heat_0" in engine.step(0, 12.0, mask).benign_instances
    assert "heat_0" not in engine.step(1, 12.0, mask).benign_instances
    assert "heat_0" not in engine.benign_instances


def test_configured_venue_defaults_are_valid_types():
    config = ConfoundersConfig()
    assert all(VenueType(value) for value in config.venue_crowding_venue_types)


def test_background_ili_secondary_probability_grows_household_clusters():
    counts = []
    household_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    for probability in (0.0, 0.5, 1.0):
        engine = ConfounderEngine(
            6,
            ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
                background_ili_daily_incidence=0.2,
                background_ili_secondary_probability=probability,
                background_ili_incubation_delay_steps=0,
                background_ili_symptomatic_duration_steps=4,
            ),
            np.random.default_rng(42),
            household_ids=household_ids,
        )
        step = engine.step(0, 12.0, np.ones(6, dtype=bool))
        counts.append(
            len(step.affected_agents_by_cause.get(PerturbationCause.BACKGROUND_ILI, set()))
        )
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_background_ili_amplitude_is_stable_within_an_instance():
    engine = ConfounderEngine(
        4,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            background_ili_daily_incidence=1.0,
            background_ili_incubation_delay_steps=0,
            background_ili_symptomatic_duration_steps=4,
            background_ili_amplitude_jitter=0.4,
        ),
        np.random.default_rng(42),
        household_ids=np.arange(4, dtype=np.int64),
    )
    mask = np.ones(4, dtype=bool)
    first = engine.step(0, 12.0, mask)
    second = engine.step(1, 12.0, mask)
    first_delta = first.contributions[0][0].delta
    second_delta = second.contributions[0][0].delta
    assert np.allclose(second_delta, first_delta * (3.0 / 4.0))


def test_background_ili_does_not_overwrite_existing_household_illness():
    household_ids = np.array([0, 0, 0], dtype=np.int64)
    engine = ConfounderEngine(
        3,
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            background_ili_daily_incidence=1.0,
            background_ili_secondary_probability=1.0,
            background_ili_incubation_delay_steps=0,
            background_ili_symptomatic_duration_steps=4,
        ),
        np.random.default_rng(42),
        household_ids=household_ids,
    )
    engine._ili_remaining[1] = 2
    step = engine.step(0, 12.0, np.ones(3, dtype=bool))
    affected = step.affected_agents_by_cause[PerturbationCause.BACKGROUND_ILI]
    assert 1 not in affected
    assert engine._ili_remaining[1] == 1


def test_confounders_are_hazard_independent():
    confounders = ConfoundersConfig(
        enabled=True,
        exercise_rate=0.1,
        sleep_disruption_rate=0.1,
        sensor_artifact_probability=0.5,
        heat_wave_start_step=4,
        heat_wave_duration_steps=8,
    )
    common = dict(
        n_agents=40,
        wearable_fraction=0.8,
        n_steps=24,
        seed=7,
        mobility_model="static",
        world_settling_steps=0,
        confounders=confounders,
    )
    hazard_free = GarlandModel(
        SimulationConfig(
            **common,
            seir=SEIRConfig(initial_infected=0),
            plumes=[],
        )
    )
    hazard_active = GarlandModel(
        SimulationConfig(
            **common,
            seir=SEIRConfig(initial_infected=4),
            plumes=[PlumeConfig(start_step=0, duration_steps=24)],
        )
    )
    hazard_free.run()
    hazard_active.run()

    assert (
        hazard_free.metrics.summary()["confounder_contributions_by_cause"]
        == hazard_active.metrics.summary()["confounder_contributions_by_cause"]
    )
    assert (
        hazard_free.metrics.summary()["confounder_agents_affected_by_cause"]
        == hazard_active.metrics.summary()["confounder_agents_affected_by_cause"]
    )


def test_exercise_rate_grades_full_pipeline_background_emission_rate():
    rates = []
    for exercise_rate in (0.0, 0.03, 0.15):
        model = GarlandModel(
            SimulationConfig(
                n_agents=50,
                wearable_fraction=0.8,
                n_steps=144,
                seed=9,
                mobility_model="static",
                world_settling_steps=0,
                confounders=ConfoundersConfig(
                    enabled=True,
                    exercise_rate=exercise_rate,
                    sleep_disruption_rate=0.0,
                    sensor_artifact_probability=0.0,
                ),
            )
        )
        model.run()
        rates.append(model.metrics.summary()["background_rate"])

    assert rates == sorted(rates)
    assert rates[-1] > rates[0]


def test_settled_family_signature_distinguishes_independent_and_shared_sources():
    def run(confounders: ConfoundersConfig) -> tuple[dict, list[int]]:
        model = GarlandModel(
            SimulationConfig(
                n_agents=120,
                wearable_fraction=0.8,
                n_steps=576,
                seed=42,
                mobility_model="static",
                world_settling_steps=144,
                confounders=confounders,
                # This grades family signatures at the calibrated operating
                # point; respondent-basis dilation is exercised separately.
                privacy=PrivacyConfig(dilation_basis="residents"),
            )
        )
        model.run()
        return model.metrics.summary(), [
            row["alarming_zones"] for row in model.metrics.step_records[144:]
        ]

    family_a, family_a_breadth = run(
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.01,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
        )
    )
    heat_wave, heat_breadth = run(
        ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
            heat_wave_start_step=288,
            heat_wave_duration_steps=96,
            heat_wave_hr_delta=12.5,
            heat_wave_temperature_delta=2.0,
            heat_wave_peak_hour=4.0,
            has_air_conditioning_fraction=0.0,
            heat_island_gain=0.75,
        )
    )

    family_a_dispersion = family_a["background_settled_window_pearson_dispersion"]
    heat_dispersion = heat_wave["background_settled_window_pearson_dispersion"]
    assert family_a_dispersion < 2.0
    assert heat_dispersion > family_a_dispersion + 0.3
    assert max(heat_breadth) > max(family_a_breadth)
    assert sum(width > 1 for width in heat_breadth) > sum(width > 1 for width in family_a_breadth)


def test_model_locality_contrast_venue_vs_heat_wave():
    def run(source: str) -> list[int]:
        venue = VenueConfig(
            venue_id="gathering",
            venue_type=VenueType.GATHERING.value,
            center_x=500.0,
            center_y=500.0,
            radius=20.0,
            capacity=30,
        )
        confounders = ConfoundersConfig(
            enabled=True,
            exercise_rate=0.0,
            sleep_disruption_rate=0.0,
            sensor_artifact_probability=0.0,
        )
        if source == "venue":
            confounders.venue_crowding_rate = 1.0
            confounders.venue_crowding_duration_steps = 12
            confounders.venue_crowding_venue_types = (VenueType.GATHERING,)
        else:
            confounders.heat_wave_start_step = 0
            confounders.heat_wave_duration_steps = 12
            confounders.has_air_conditioning_fraction = 0.0
            confounders.heat_wave_hr_delta = 12.5
            confounders.heat_wave_temperature_delta = 2.0
            confounders.heat_wave_peak_hour = 3.0
            confounders.heat_island_gain = 0.75
        model = GarlandModel(
            SimulationConfig(
                n_agents=120,
                wearable_fraction=0.8,
                n_steps=96,
                seed=42,
                mobility_model="static",
                world_settling_steps=24,
                seir=SEIRConfig(initial_infected=0),
                plumes=[],
                venues=VenueSystemConfig(
                    enabled=True,
                    venues=[venue],
                    position_jitter_fraction=0.0,
                ),
                confounders=confounders,
                # This grades locality at the calibrated operating point;
                # respondent-basis dilation is exercised separately.
                privacy=PrivacyConfig(dilation_basis="residents"),
            )
        )
        cells = model.agent_cell_ids.copy()
        cell_values, cell_counts = np.unique(cells, return_counts=True)
        target_cell = int(cell_values[np.argmax(cell_counts)])
        venue_agents = np.flatnonzero(cells == target_cell)
        model.venue_engine.current_venue_idx[:] = -1
        model.venue_engine.current_venue_idx[venue_agents] = 0
        model.run()
        return [row["alarming_zones"] for row in model.metrics.step_records[24:36]]

    venue_breadth = run("venue")
    heat_breadth = run("heat")
    assert max(heat_breadth) >= max(venue_breadth) + 3
    assert max(heat_breadth) >= 6


def test_disabled_confounders_do_not_change_hazard_metrics():
    def run(confounders: ConfoundersConfig | None) -> tuple[dict, GarlandModel]:
        model = GarlandModel(
            SimulationConfig(
                n_agents=80,
                wearable_fraction=0.8,
                n_steps=36,
                seed=17,
                mobility_model="static",
                world_settling_steps=0,
                seir=SEIRConfig(initial_infected=4),
                plumes=[PlumeConfig(start_step=4, duration_steps=20)],
                **({"confounders": confounders} if confounders is not None else {}),
            )
        )
        model.run()
        return model.metrics.summary(), model

    baseline, baseline_model = run(None)
    disabled, disabled_model = run(ConfoundersConfig(enabled=False))
    for key in (
        "fpr_disease",
        "fpr_toxin",
        "discrimination_score",
        "detection_event_counts",
    ):
        assert disabled[key] == baseline[key]
    for attribute in (
        "false_positives_disease",
        "false_positives_toxin",
        "true_negatives_disease",
        "true_negatives_toxin",
        "false_negatives_disease",
        "false_negatives_toxin",
        "true_positives_disease",
        "true_positives_toxin",
    ):
        assert getattr(disabled_model.metrics, attribute) == getattr(
            baseline_model.metrics, attribute
        )


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

    # This enabled-with-zero-sources comparison verifies confounder RNG isolation.
    for key in (
        "total_broadcasts",
        "total_responses",
        "total_epsilon",
        "background_rate",
    ):
        assert disabled_summary[key] == zero_summary[key]
    assert zero_summary["confounder_contributions_by_cause"] == {}
    assert zero_summary["heat_wave_active_steps"] == 0


def test_benign_scoring_conserves_hazards_off():
    model = GarlandModel(
        SimulationConfig(
            n_agents=20,
            wearable_fraction=0.8,
            n_steps=24,
            seed=1,
            mobility_model="static",
            world_settling_steps=0,
            seir=SEIRConfig(initial_infected=0),
            plumes=[],
            # This grades confounder warrants at the calibrated operating
            # point; respondent-basis dilation is exercised separately.
            # The scenario needs a satisfiable anonymity bound for a
            # legitimate broadcast to exist at all.
            privacy=PrivacyConfig(k_min=10, dilation_basis="residents"),
            confounders=ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
                heat_wave_duration_steps=24,
                heat_wave_hr_delta=20.0,
                heat_wave_temperature_delta=3.0,
                heat_wave_peak_hour=1.0,
                has_air_conditioning_fraction=0.0,
            ),
        )
    )
    model.run()
    summary = model.metrics.summary()
    overlap = summary["benign_overlap_detections"]
    attributed = summary["benign_attributed_detections"]
    misattributed = summary["benign_misattributed_detections"]
    total = len(model.metrics.detection_events)
    assert misattributed > 0
    assert misattributed <= attributed <= overlap <= total
    assert 0.0 <= summary["benign_misattribution_rate"] <= 1.0


def _assert_warrant_conservation(summary: dict) -> None:
    classes = (
        "target_detections",
        "actionable_non_target_detections",
        "explained_detections",
        "artifact_detections",
        "unexplained_detections",
    )
    total_events = summary["total_detection_events"]
    assert total_events == sum(summary[key] for key in classes)


@pytest.mark.parametrize("spatial_backend", ["hex", "rect"])
def test_heat_warrants_and_affected_subset_work_on_both_backends(spatial_backend: str):
    model = GarlandModel(
        SimulationConfig(
            n_agents=80,
            wearable_fraction=0.5,
            n_steps=12,
            seed=23,
            mobility_model="static",
            world_settling_steps=0,
            spatial_backend=spatial_backend,
            seir=SEIRConfig(initial_infected=0),
            plumes=[],
            confounders=ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
                heat_wave_duration_steps=12,
                heat_wave_peak_hour=1.0,
                has_air_conditioning_fraction=0.5,
                heat_wave_ac_exposure_multiplier=0.0,
            ),
        )
    )
    wearable_agents = set(np.flatnonzero(model.has_wearable))
    for _ in range(12):
        model.step()
        affected = model._confounder_step.affected_agents_by_cause.get(
            PerturbationCause.HEAT_WAVE, set()
        )
        assert affected < wearable_agents
    _assert_warrant_conservation(model.metrics.summary())


def test_model_warrant_classes_conserve_for_hazard_and_confounder_runs():
    hazard_model = GarlandModel(
        SimulationConfig(
            n_agents=80,
            wearable_fraction=0.8,
            n_steps=36,
            seed=17,
            mobility_model="static",
            world_settling_steps=0,
            seir=SEIRConfig(initial_infected=4),
            plumes=[PlumeConfig(start_step=4, duration_steps=20)],
            confounders=ConfoundersConfig(enabled=False),
        )
    )
    hazard_model.run()
    hazard_summary = hazard_model.metrics.summary()
    assert hazard_summary["target_detections"] > 0
    assert hazard_summary["actionable_non_target_detections"] == 0
    _assert_warrant_conservation(hazard_summary)

    confounder_model = GarlandModel(
        SimulationConfig(
            n_agents=40,
            wearable_fraction=0.8,
            n_steps=24,
            seed=18,
            mobility_model="static",
            world_settling_steps=0,
            seir=SEIRConfig(initial_infected=0),
            plumes=[],
            # The scenario needs a satisfiable anonymity bound for a
            # legitimate broadcast to exercise the warrant classes.
            privacy=PrivacyConfig(k_min=10, dilation_basis="residents"),
            confounders=ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
                heat_wave_duration_steps=24,
                heat_wave_peak_hour=1.0,
                has_air_conditioning_fraction=0.0,
                heat_wave_hr_delta=20.0,
                heat_wave_temperature_delta=3.0,
            ),
        )
    )
    confounder_model.run()
    confounder_summary = confounder_model.metrics.summary()
    assert confounder_summary["target_detections"] == 0
    _assert_warrant_conservation(confounder_summary)


def test_heat_warrants_do_not_make_all_non_targets_actionable():
    model = GarlandModel(
        SimulationConfig(
            n_agents=80,
            wearable_fraction=0.8,
            n_steps=48,
            seed=19,
            mobility_model="static",
            world_settling_steps=0,
            seir=SEIRConfig(initial_infected=0),
            plumes=[],
            # This grades confounder warrants at the calibrated operating
            # point; respondent-basis dilation is exercised separately.
            privacy=PrivacyConfig(dilation_basis="residents"),
            confounders=ConfoundersConfig(
                enabled=True,
                exercise_rate=0.4,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
                heat_wave_duration_steps=48,
                heat_wave_peak_hour=1.0,
                has_air_conditioning_fraction=1.0,
                heat_wave_hr_delta=20.0,
                heat_wave_temperature_delta=3.0,
            ),
        )
    )
    model.run()
    summary = model.metrics.summary()
    non_target = summary["total_detection_events"] - summary["target_detections"]
    assert non_target > summary["actionable_non_target_detections"]
    assert summary["explained_detections"] + summary["unexplained_detections"] > 0
    _assert_warrant_conservation(summary)


def test_onboarding_benign_instance_keeps_cohort_identity():
    model = GarlandModel(
        SimulationConfig(
            n_agents=20,
            wearable_fraction=0.5,
            n_steps=3,
            seed=4,
            mobility_model="static",
            adoption=AdoptionConfig(
                mode="trickle",
                start_step=0,
                rate=1.0,
                initial_adopted_fraction=0.0,
                onboarding_window_steps=6,
            ),
            confounders=ConfoundersConfig(
                enabled=True,
                exercise_rate=0.0,
                sleep_disruption_rate=0.0,
                sensor_artifact_probability=0.0,
            ),
        )
    )
    model.step()
    first_ids = set(model._confounder_step.benign_instances)
    model.step()
    second_ids = set(model._confounder_step.benign_instances)
    assert first_ids == second_ids
    assert first_ids
    assert all(instance_id.startswith("onboarding_step_") for instance_id in first_ids)


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
