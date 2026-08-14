"""Tests for second-round contextual disambiguation queries."""

import numpy as np
import pytest

from garland.adoption import AdoptionConfig
from garland.agents import CitizenAgent, NetworkAggregator
from garland.attacks import AttackConfig, AttackType
from garland.config import config_from_dict, config_to_dict
from garland.disambiguation import DisambiguationConfig, DisambiguationHypothesis
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
    assert model._process_disambiguation_queries([query], 0)["queries"] == 1

    model.current_step = 12
    assert model._process_disambiguation_queries([], 0)["unanswered"] == 0
    model.current_step = 13
    assert model._process_disambiguation_queries([], 0)["unanswered"] == 1


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
    assert config.hypothesis is DisambiguationHypothesis.RECENT_ADOPTION


def test_disambiguation_config_round_trips() -> None:
    config = config_from_dict(
        {
            "n_agents": 10,
            "disambiguation": {
                "enabled": True,
                "answer_rate": 0.25,
                "yes_rate": 0.8,
                "expiry_steps": 4,
                "ack_epsilon": 0.02,
            },
        }
    )
    serialized = config_to_dict(config)

    assert config.disambiguation.enabled is True
    assert serialized["disambiguation"]["answer_rate"] == 0.25
    assert serialized["disambiguation"]["ack_epsilon"] == 0.02


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

    assert result["queries"] == 1
    assert result["acks"] == 1
    assert result["yes"] + result["no"] == 1


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

    assert no_ack["acks"] == 0
    assert ack_no_answer["acks"] == 1
    assert no_ack["acks"] <= no_ack["reached"]
    assert ack_no_answer["acks"] <= ack_no_answer["reached"]
    assert ack_no_answer["yes"] == 0
    assert ack_no_answer["no"] == 0
    assert expired["unanswered"] == 1
    assert (
        ack_no_answer["yes"]
        + ack_no_answer["no"]
        + expired["unanswered"]
        == ack_no_answer["acks"]
    )


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
                answer_rate=1.0,
                yes_rate=1.0,
                min_onboarding_wearables_in_zone=1,
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
    assert (
        summary["disambiguation_acks"]
        <= summary["disambiguation_devices_reached"]
    )
    assert (
        summary["disambiguation_yes_answers"]
        + summary["disambiguation_no_answers"]
        + summary["disambiguation_unanswered_expired"]
        == summary["disambiguation_acks"]
    )
    assert summary["disambiguation_answer_epsilon"] > 0
    assert summary["disambiguation_ack_epsilon"] > 0


def test_disabled_disambiguation_summary_is_zero_and_legacy_metrics_match() -> None:
    disabled = GarlandModel(_integrated_config(DisambiguationConfig())).run().summary()
    explicit = GarlandModel(
        _integrated_config(DisambiguationConfig(enabled=False))
    ).run().summary()

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
        assert explicit[key] == 0

    legacy_keys = (
        "total_broadcasts",
        "total_responses",
        "total_epsilon",
        "fleet_cold_start",
        "fleet_cold_baseline_wearable_step_fraction",
        "post_world_settling_cold_baseline_wearable_step_fraction",
        "adoption_events",
    )
    for key in legacy_keys:
        assert disabled[key] == explicit[key]
