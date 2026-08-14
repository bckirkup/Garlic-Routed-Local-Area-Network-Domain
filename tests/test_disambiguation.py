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
        1,
        DisambiguationHypothesis.RECENT_ADOPTION,
        expiry_steps=2,
        should_ask=lambda _: True,
    )
    assert len(queries) == 1
    aggregator.pending_disambiguation[queries[0].query_id] = (3, 2, False)

    assert aggregator.expire_disambiguation(2) == (0, 0)
    assert aggregator.expire_disambiguation(3) == (2, 1)


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
    expired = declining._process_disambiguation_queries([], 2)

    assert no_ack["acks"] == 0
    assert ack_no_answer["acks"] == 1
    assert ack_no_answer["yes"] == 0
    assert ack_no_answer["no"] == 0
    assert expired["unanswered"] == 1
