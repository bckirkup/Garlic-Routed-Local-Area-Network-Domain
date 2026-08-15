"""Tests for second-round contextual disambiguation queries."""

import numpy as np
import pytest

from garland.adoption import AdoptionConfig
from garland.agents import CitizenAgent, NetworkAggregator
from garland.attacks import AttackConfig, AttackType
from garland.config import config_from_dict, config_to_dict
from garland.confounders import BenignInstance, ConfounderStep
from garland.disambiguation import (
    DisambiguationConfig,
    DisambiguationHypothesis,
    DisambiguationTriggerConfig,
)
from garland.perturbations import PerturbationCause
from garland.privacy import (
    AggregatorState,
    AnomalyType,
    BroadcastQuery,
    DisambiguationQuery,
    PrivacyConfig,
)
from garland.simulation import GarlandModel, SimulationConfig


def _query() -> DisambiguationQuery:
    return DisambiguationQuery(
        zone_cells=[1],
        hypothesis=DisambiguationHypothesis.RECENT_ADOPTION,
        referenced_query_ids=(3,),
        time_window_start=0,
        time_window_end=4,
    )


def test_disambiguation_ack_is_separate_from_human_answer() -> None:
    agent = CitizenAgent(idx=0, has_wearable=True)
    query = _query()
    privacy = PrivacyConfig(randomized_response_p=1.0)

    no_answer = agent.respond_to_disambiguation(
        query, 1, answer_rate=0.0, yes_rate=1.0, config=privacy, rng=np.random.default_rng(1)
    )
    answer_yes = agent.respond_to_disambiguation(
        query, 1, answer_rate=1.0, yes_rate=1.0, config=privacy, rng=np.random.default_rng(1)
    )

    assert no_answer is None
    assert answer_yes is True


def test_disambiguation_expiry_counts_unanswered_as_unresolved() -> None:
    aggregator = NetworkAggregator(config=PrivacyConfig(k_min=1))
    broadcast = BroadcastQuery(
        zone_cells=[1],
        anomaly_type=AnomalyType.FEBRILE,
        time_window_start=0,
        time_window_end=1,
        query_id=4,
    )
    queries = aggregator.issue_disambiguation_queries(
        [broadcast],
        DisambiguationHypothesis.RECENT_ADOPTION,
        should_ask=lambda _: True,
    )
    assert len(queries) == 1
    aggregator.register_disambiguation_pending(
        queries[0].query_id,
        expires_at_step=3,
        unanswered_count=2,
        answered=False,
    )

    assert aggregator.expire_disambiguation(2) == (0, 0)
    assert aggregator.expire_disambiguation(3) == (2, 1)


def test_disambiguation_expiry_horizon_is_measured_in_steps() -> None:
    model = GarlandModel(
        SimulationConfig(
            n_agents=20,
            n_steps=1,
            wearable_fraction=0.5,
            mobility_model="static",
            spatial_backend="rect",
            grid_width=1000.0,
            grid_height=1000.0,
            cell_size=200.0,
            adoption=AdoptionConfig(),
            disambiguation=DisambiguationConfig(
                enabled=True,
                enabled_hypotheses=frozenset({DisambiguationHypothesis.RECENT_ADOPTION}),
                recent_adoption=DisambiguationTriggerConfig(
                    max_zone_cells=100,
                    min_persistent_windows=1,
                ),
                answer_rate=0.0,
                expiry_steps=3,
            ),
            privacy=PrivacyConfig(k_min=1),
        )
    )
    agent = model.citizen_agents[0]
    agent.fleet_start_adopter = False
    agent.adoption_step = 10
    agent.steps_since_adoption = 0
    model.wearable_agents_by_cell = {agent.cell_id: [agent]}
    query = BroadcastQuery(
        zone_cells=[agent.cell_id],
        anomaly_type=AnomalyType.FEBRILE,
        time_window_start=0,
        time_window_end=1,
    )
    model.current_step = 10
    assert model._process_disambiguation_queries([query], 0).queries == 1

    model.current_step = 12
    assert model._process_disambiguation_queries([], 0).unanswered == 0
    model.current_step = 13
    assert model._process_disambiguation_queries([], 0).unanswered == 1


def test_disambiguation_answer_rate_changes_approved_answer_count() -> None:
    agent = CitizenAgent(idx=0, has_wearable=True)
    query = _query()
    privacy = PrivacyConfig(randomized_response_p=1.0)
    counts = []
    for rate in (0.0, 0.5, 1.0):
        rng = np.random.default_rng(11)
        counts.append(
            sum(
                agent.respond_to_disambiguation(
                    query,
                    1,
                    answer_rate=rate,
                    yes_rate=1.0,
                    config=privacy,
                    rng=rng,
                )
                is not None
                for _ in range(1000)
            )
        )
    assert counts[0] == 0
    assert counts[0] < counts[1] < counts[2]


def test_disambiguation_config_defaults_disabled() -> None:
    config = DisambiguationConfig()
    assert config.enabled is False
    assert config.enabled_hypotheses == frozenset()


def test_enabled_disambiguation_requires_hypotheses() -> None:
    with pytest.raises(ValueError, match="requires at least one enabled hypothesis"):
        DisambiguationConfig(enabled=True)


def test_disambiguation_config_round_trips() -> None:
    config = config_from_dict(
        {
            "n_agents": 10,
            "disambiguation": {
                "enabled": True,
                "enabled_hypotheses": ["recent_adoption", "ambient_heat"],
                "recent_adoption": {"max_zone_cells": 2},
                "answer_rate": 0.25,
                "yes_rate": 0.8,
                "expiry_steps": 4,
                "ack_epsilon": 0.02,
            },
        }
    )
    serialized = config_to_dict(config)

    assert config.disambiguation.enabled is True
    assert config.disambiguation.enabled_hypotheses == frozenset(
        {DisambiguationHypothesis.RECENT_ADOPTION, DisambiguationHypothesis.AMBIENT_HEAT}
    )
    assert config.disambiguation.recent_adoption.max_zone_cells == 2
    assert serialized["disambiguation"]["answer_rate"] == pytest.approx(0.25)
    assert serialized["disambiguation"]["ack_epsilon"] == pytest.approx(0.02)


def test_yes_and_no_answers_both_charge_epsilon() -> None:
    state = AggregatorState()
    state.record_disambiguation_answers(1, 0.1)
    yes_epsilon = state.disambiguation_answer_epsilon
    state.record_disambiguation_answers(1, 0.1)

    assert yes_epsilon > 0
    assert state.disambiguation_answer_count == 2
    assert state.disambiguation_answer_epsilon > yes_epsilon


@pytest.mark.parametrize("backend", ["rect", "hex"])
def test_disambiguation_predicate_works_on_both_spatial_backends(
    backend: str,
) -> None:
    config = SimulationConfig(
        n_agents=20,
        n_steps=1,
        wearable_fraction=0.5,
        mobility_model="static",
        spatial_backend=backend,
        grid_width=1000.0,
        grid_height=1000.0,
        cell_size=200.0,
        adoption=AdoptionConfig(),
        disambiguation=DisambiguationConfig(
            enabled=True,
            enabled_hypotheses=frozenset({DisambiguationHypothesis.RECENT_ADOPTION}),
            recent_adoption=DisambiguationTriggerConfig(min_persistent_windows=1),
            answer_rate=1.0,
            yes_rate=1.0,
        ),
        privacy=PrivacyConfig(k_min=1),
    )
    model = GarlandModel(config)
    agent = model.citizen_agents[0]
    agent.fleet_start_adopter = False
    agent.adoption_step = 0
    agent.steps_since_adoption = 0
    model.wearable_agents_by_cell = {agent.cell_id: [agent]}
    broadcast = BroadcastQuery(
        zone_cells=[agent.cell_id],
        anomaly_type=AnomalyType.FEBRILE,
        time_window_start=0,
        time_window_end=1,
        query_id=9,
    )

    result = model._process_disambiguation_queries([broadcast], 1)

    assert result.queries == 1
    assert result.acks == 1
    assert result.yes + result.no == 1


def test_eclipsed_zone_has_no_ack_but_declining_population_does() -> None:
    def make_model(active_attacks: list[AttackType], answer_rate: float) -> GarlandModel:
        model = GarlandModel(
            SimulationConfig(
                n_agents=20,
                n_steps=1,
                wearable_fraction=0.5,
                mobility_model="static",
                spatial_backend="rect",
                grid_width=1000.0,
                grid_height=1000.0,
                cell_size=200.0,
                adoption=AdoptionConfig(),
                disambiguation=DisambiguationConfig(
                    enabled=True,
                    enabled_hypotheses=frozenset({DisambiguationHypothesis.RECENT_ADOPTION}),
                    recent_adoption=DisambiguationTriggerConfig(
                        max_zone_cells=100,
                        min_persistent_windows=1,
                    ),
                    answer_rate=answer_rate,
                    expiry_steps=1,
                ),
                privacy=PrivacyConfig(k_min=1),
                attacks=AttackConfig(
                    active_attacks=active_attacks,
                    eclipse_target_zones=[],
                    eclipse_drop_fraction=1.0,
                ),
            )
        )
        agent = model.citizen_agents[0]
        agent.fleet_start_adopter = False
        agent.adoption_step = 0
        agent.steps_since_adoption = 0
        model.wearable_agents_by_cell = {agent.cell_id: [agent]}
        return model

    eclipsed = make_model([AttackType.ECLIPSE], answer_rate=1.0)
    eclipsed.config.attacks.eclipse_target_zones = [eclipsed.citizen_agents[0].cell_id]
    declining = make_model([], answer_rate=0.0)
    query = BroadcastQuery(
        zone_cells=[eclipsed.citizen_agents[0].cell_id],
        anomaly_type=AnomalyType.FEBRILE,
        time_window_start=0,
        time_window_end=1,
    )

    no_ack = eclipsed._process_disambiguation_queries([query], 1)
    ack_no_answer = declining._process_disambiguation_queries([query], 1)
    declining.current_step = 2
    expired = declining._process_disambiguation_queries([], 2)

    assert no_ack.acks == 0
    assert ack_no_answer.acks == 1
    assert no_ack.acks <= no_ack.reached
    assert ack_no_answer.acks <= ack_no_answer.reached
    assert ack_no_answer.yes == 0
    assert ack_no_answer.no == 0
    assert expired.unanswered == 1
    assert ack_no_answer.yes + ack_no_answer.no + expired.unanswered == ack_no_answer.acks


def _shape_model(
    *,
    max_zone_cells: int = 3,
    min_persistent_windows: int = 1,
    max_confirmed_fraction: float = 0.5,
    min_breadth: int = 4,
    hypotheses: frozenset[DisambiguationHypothesis] | None = None,
) -> GarlandModel:
    return GarlandModel(
        SimulationConfig(
            n_agents=12,
            n_steps=2,
            wearable_fraction=1.0,
            mobility_model="static",
            spatial_backend="rect",
            grid_width=1000.0,
            grid_height=1000.0,
            cell_size=200.0,
            disambiguation=DisambiguationConfig(
                enabled=True,
                enabled_hypotheses=hypotheses
                or frozenset({DisambiguationHypothesis.RECENT_ADOPTION}),
                recent_adoption=DisambiguationTriggerConfig(
                    max_zone_cells=max_zone_cells,
                    min_persistent_windows=min_persistent_windows,
                    max_confirmed_fraction=max_confirmed_fraction,
                ),
                ambient_heat=DisambiguationTriggerConfig(min_breadth=min_breadth),
                answer_rate=0.0,
            ),
            privacy=PrivacyConfig(k_min=1, randomized_response_p=1.0),
        )
    )


def _broadcast(zone_cells: list[int], query_id: int = 0) -> BroadcastQuery:
    return BroadcastQuery(
        zone_cells=zone_cells,
        anomaly_type=AnomalyType.FEBRILE,
        time_window_start=0,
        time_window_end=1,
        query_id=query_id,
    )


def test_disambiguation_trigger_does_not_read_onboarding_state() -> None:
    model = _shape_model(max_zone_cells=1)
    agent = model.citizen_agents[0]
    agent.fleet_start_adopter = False
    agent.adoption_step = 0
    agent.steps_since_adoption = 0

    result = model._process_disambiguation_queries(
        [_broadcast([agent.cell_id, agent.cell_id + 1, agent.cell_id + 2, agent.cell_id + 3])],
        0,
    )

    assert result.queries == 0


def test_right_shape_without_onboarding_is_unfounded() -> None:
    model = _shape_model()
    agent = model.citizen_agents[0]
    model.wearable_agents_by_cell = {agent.cell_id: [agent]}
    model._confounder_step = ConfounderStep(
        contributions={},
        affected_agents_by_cause={},
        benign_instances={
            "heat_0": BenignInstance(
                instance_id="heat_0",
                cause=PerturbationCause.HEAT_WAVE,
                start_step=0,
                end_step=2,
                global_scope=True,
            )
        },
    )

    result = model._process_disambiguation_queries([_broadcast([agent.cell_id])], 0)

    assert result.queries == 1
    assert result.well_founded == 0
    assert result.unfounded == 1
    assert result.unfounded_by_hypothesis == {"recent_adoption": 1}
    assert result.unfounded_epsilon > 0


def test_disambiguation_without_benign_ground_truth_is_unscored() -> None:
    model = _shape_model()
    agent = model.citizen_agents[0]
    model.wearable_agents_by_cell = {agent.cell_id: [agent]}

    result = model._process_disambiguation_queries([_broadcast([agent.cell_id])], 0)

    assert result.queries == 1
    assert result.well_founded == 0
    assert result.unfounded == 0
    assert result.unscored == 1
    assert result.unfounded_epsilon == pytest.approx(0.0)
    assert result.unscored_epsilon > 0
    assert result.unscored_by_hypothesis == {"recent_adoption": 1}


def test_disambiguation_thresholds_change_ask_counts_monotonically() -> None:
    def run(**kwargs: object) -> int:
        model = _shape_model(**kwargs)
        agent = model.citizen_agents[0]
        model.wearable_agents_by_cell = {agent.cell_id: [agent]}
        return model._process_disambiguation_queries([_broadcast([agent.cell_id])], 0).queries

    assert run(min_persistent_windows=1) > run(min_persistent_windows=2)
    assert run(max_zone_cells=0) == 0


def test_disambiguation_persistence_uses_trigger_cell_identity() -> None:
    model = _shape_model(min_persistent_windows=2, max_zone_cells=3)
    first = _broadcast([1, 2], query_id=0)
    second = BroadcastQuery(
        zone_cells=[2, 3],
        anomaly_type=AnomalyType.FEBRILE,
        time_window_start=1,
        time_window_end=2,
        query_id=1,
    )
    first.trigger_cell_id = 7
    second.trigger_cell_id = 7

    model._update_disambiguation_history([first], 0)
    model._update_disambiguation_history([second], 1)

    assert model._disambiguation_worthwhile(
        second,
        DisambiguationHypothesis.RECENT_ADOPTION,
        model.config.disambiguation.recent_adoption,
        breadth=1,
    )


def test_ambient_heat_asks_increase_with_simultaneous_breadth() -> None:
    def run(min_breadth: int) -> int:
        model = _shape_model(
            min_breadth=min_breadth,
            hypotheses=frozenset({DisambiguationHypothesis.AMBIENT_HEAT}),
        )
        queries = [_broadcast([cell], query_id=cell) for cell in range(4)]
        return model._process_disambiguation_queries(queries, 0).queries

    assert run(3) > run(5)


def test_disambiguation_scoring_conserves_well_founded_and_unfounded() -> None:
    model = _shape_model(
        hypotheses=frozenset(
            {DisambiguationHypothesis.RECENT_ADOPTION, DisambiguationHypothesis.AMBIENT_HEAT}
        ),
        min_breadth=2,
    )
    agent = model.citizen_agents[0]
    model.wearable_agents_by_cell = {agent.cell_id: [agent]}
    model._confounder_step = ConfounderStep(
        contributions={},
        affected_agents_by_cause={},
        benign_instances={
            "heat_0": BenignInstance(
                instance_id="heat_0",
                cause=PerturbationCause.HEAT_WAVE,
                start_step=0,
                end_step=2,
                global_scope=True,
            )
        },
    )
    queries = [_broadcast([agent.cell_id], query_id=0), _broadcast([agent.cell_id + 1], 1)]

    result = model._process_disambiguation_queries(queries, 0)

    assert result.well_founded + result.unfounded + result.unscored == result.queries
    assert result.well_founded_by_hypothesis == {"ambient_heat": 1}
    assert result.unfounded_by_hypothesis == {"recent_adoption": 1}


def test_disambiguation_disabled_hazard_metrics_do_not_change() -> None:
    base = _integrated_config(DisambiguationConfig())
    disabled = GarlandModel(base).run().summary()
    explicit = GarlandModel(_integrated_config(DisambiguationConfig(enabled=False))).run().summary()

    for key in (
        "fpr_disease",
        "fpr_toxin",
        "discrimination_score",
        "detection_event_counts",
    ):
        assert explicit[key] == disabled[key]


def _integrated_config(disambiguation: DisambiguationConfig) -> SimulationConfig:
    return SimulationConfig(
        n_agents=40,
        n_steps=6,
        wearable_fraction=0.5,
        mobility_model="static",
        spatial_backend="rect",
        grid_width=1000.0,
        grid_height=1000.0,
        cell_size=200.0,
        anomaly_threshold=-1.0,
        baseline_warmup_steps=0,
        privacy=PrivacyConfig(
            threshold_m=1,
            k_min=1,
            randomized_response_p=1.0,
        ),
        adoption=AdoptionConfig(
            mode="cohort",
            initial_adopted_fraction=0.5,
            start_step=1,
            cohort_size=2,
            interval_steps=1,
            group_by="household",
        ),
        disambiguation=disambiguation,
    )


def test_enabled_disambiguation_runs_through_model_and_preserves_invariants() -> None:
    model = GarlandModel(
        _integrated_config(
            DisambiguationConfig(
                enabled=True,
                enabled_hypotheses=frozenset({DisambiguationHypothesis.RECENT_ADOPTION}),
                recent_adoption=DisambiguationTriggerConfig(
                    max_zone_cells=100,
                    min_persistent_windows=1,
                    max_confirmed_fraction=1.0,
                ),
                answer_rate=1.0,
                yes_rate=1.0,
                expiry_steps=2,
                ack_noise_scale=0.0,
            )
        )
    )
    summary = model.run().summary()

    assert summary["disambiguation_queries_issued"] > 0
    assert summary["disambiguation_acks"] > 0
    assert summary["disambiguation_yes_answers"] > 0
    assert summary["disambiguation_no_answers"] == 0
    assert summary["disambiguation_unanswered_expired"] == 0
    assert summary["disambiguation_unresolved_hypotheses"] == 0
    assert summary["disambiguation_acks"] <= summary["disambiguation_devices_reached"]
    assert (
        summary["disambiguation_yes_answers"]
        + summary["disambiguation_no_answers"]
        + summary["disambiguation_unanswered_expired"]
        == summary["disambiguation_acks"]
    )
    assert summary["disambiguation_answer_epsilon"] > 0
    assert summary["disambiguation_ack_epsilon"] > 0


def test_disambiguation_is_additive_without_moving_round_one_metrics() -> None:
    disabled = GarlandModel(_integrated_config(DisambiguationConfig())).run().summary()
    enabled = (
        GarlandModel(
            _integrated_config(
                DisambiguationConfig(
                    enabled=True,
                    enabled_hypotheses=frozenset({DisambiguationHypothesis.RECENT_ADOPTION}),
                    recent_adoption=DisambiguationTriggerConfig(
                        max_zone_cells=100,
                        min_persistent_windows=1,
                        max_confirmed_fraction=1.0,
                    ),
                    answer_rate=1.0,
                    yes_rate=1.0,
                    ack_noise_scale=0.0,
                )
            )
        )
        .run()
        .summary()
    )

    disambiguation_keys = (
        "disambiguation_queries_issued",
        "disambiguation_acks",
        "disambiguation_ack_release_count",
        "disambiguation_devices_reached",
        "disambiguation_yes_answers",
        "disambiguation_no_answers",
        "disambiguation_unanswered_expired",
        "disambiguation_unresolved_hypotheses",
        "disambiguation_answer_epsilon",
        "disambiguation_ack_epsilon",
    )
    for key in disambiguation_keys:
        assert disabled[key] == 0
    assert enabled["disambiguation_queries_issued"] > 0
    assert enabled["disambiguation_acks"] > 0
    assert enabled["disambiguation_yes_answers"] > 0
    assert enabled["disambiguation_answer_epsilon"] > 0
    assert enabled["disambiguation_ack_epsilon"] > 0

    legacy_keys = (
        "total_broadcasts",
        "total_responses",
        "fleet_cold_start",
        "fleet_cold_baseline_wearable_step_fraction",
        "post_world_settling_cold_baseline_wearable_step_fraction",
        "adoption_events",
    )
    for key in legacy_keys:
        assert disabled[key] == enabled[key]
    enabled_round_one_epsilon = (
        enabled["total_epsilon"]
        - enabled["disambiguation_answer_epsilon"]
        - enabled["disambiguation_ack_epsilon"]
    )
    assert enabled_round_one_epsilon == pytest.approx(disabled["total_epsilon"])
