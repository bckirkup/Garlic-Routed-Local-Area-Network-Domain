"""Unit tests for the GARLAND attack layer and orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from garland.attacks import (
    AttackConfig,
    AttackOrchestrator,
    AttackType,
    EclipseAttacker,
    ReplayAttacker,
)
from garland.privacy import AnomalyType, EncryptedToken, PerturbedResponse


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestEclipseAttacker:
    """Eclipse intercepts tokens from target zones."""

    def test_drops_tokens_in_target_zone(self, rng):
        attacker = EclipseAttacker()
        token = EncryptedToken(
            zone_id=5,
            anomaly_type=AnomalyType.RESPIRATORY,
            timestamp_bin=1,
            agent_id_hash=1,
            is_dummy=False,
        )
        result = attacker.intercept_token(token, {5}, rng, drop_fraction=1.0)
        assert result is None
        assert attacker.dropped_count == 1

    def test_passes_tokens_outside_target_zone(self, rng):
        attacker = EclipseAttacker()
        token = EncryptedToken(
            zone_id=3,
            anomaly_type=AnomalyType.FEBRILE,
            timestamp_bin=1,
            agent_id_hash=1,
            is_dummy=False,
        )
        result = attacker.intercept_token(token, {5}, rng, drop_fraction=1.0)
        assert result is token
        assert attacker.dropped_count == 0

    def test_partial_drop_fraction(self, rng):
        attacker = EclipseAttacker()
        token = EncryptedToken(
            zone_id=5,
            anomaly_type=AnomalyType.RESPIRATORY,
            timestamp_bin=1,
            agent_id_hash=1,
            is_dummy=False,
        )
        dropped = 0
        passed = 0
        for _ in range(200):
            result = attacker.intercept_token(token, {5}, rng, drop_fraction=0.5)
            if result is None:
                dropped += 1
            else:
                passed += 1
        assert dropped > 0
        assert passed > 0


class TestReplayAttacker:
    """Replay re-injects cached stale tokens."""

    def test_replay_injects_on_interval(self, rng):
        config = AttackConfig(
            replay_interval_steps=6,
            replay_lag_bins=0,
            replay_count=3,
            active_attacks=[AttackType.REPLAY],
        )
        attacker = ReplayAttacker(config=config)
        attacker.cache_tokens(
            [
                EncryptedToken(
                    zone_id=2,
                    anomaly_type=AnomalyType.RESPIRATORY,
                    timestamp_bin=0,
                    agent_id_hash=10,
                    is_dummy=False,
                )
            ]
        )
        tokens = attacker.generate_replay_tokens(current_step=6, time_bin=1, rng=rng)
        assert len(tokens) == 1
        assert tokens[0].zone_id == 2
        assert tokens[0].timestamp_bin == 1
        assert attacker.replay_inject_count == 1

    def test_replay_skips_dummy_tokens_in_cache(self, rng):
        config = AttackConfig(
            replay_interval_steps=6,
            replay_lag_bins=0,
            replay_count=5,
            active_attacks=[AttackType.REPLAY],
        )
        attacker = ReplayAttacker(config=config)
        attacker.cache_tokens(
            [
                EncryptedToken(
                    zone_id=1,
                    anomaly_type=AnomalyType.FEBRILE,
                    timestamp_bin=0,
                    agent_id_hash=1,
                    is_dummy=True,
                )
            ]
        )
        assert attacker.token_cache == []


class TestAttackOrchestrator:
    """Orchestrator coordinates filtering, injection, and observation."""

    def test_filter_tokens_counts_drops(self, rng):
        config = AttackConfig(
            eclipse_target_zones=[7],
            active_attacks=[AttackType.ECLIPSE],
        )
        orch = AttackOrchestrator(config=config)
        tokens = [
            EncryptedToken(7, AnomalyType.RESPIRATORY, 0, 1, False),
            EncryptedToken(8, AnomalyType.FEBRILE, 0, 2, False),
        ]
        filtered, dropped = orch.filter_tokens(tokens, rng)
        assert dropped == 1
        assert len(filtered) == 1
        assert filtered[0].zone_id == 8

    def test_step_injections_sybil_and_replay(self, rng):
        config = AttackConfig(
            sybil_count=5,
            sybil_target_zone=3,
            replay_interval_steps=6,
            replay_lag_bins=0,
            replay_count=2,
            active_attacks=[AttackType.SYBIL_INJECTION, AttackType.REPLAY],
        )
        orch = AttackOrchestrator(config=config)
        orch.replay.cache_tokens(
            [
                EncryptedToken(4, AnomalyType.RESPIRATORY, 0, 9, False),
                EncryptedToken(4, AnomalyType.FEBRILE, 0, 8, False),
            ]
        )
        injected, sybil_n, replay_n = orch.step_injections(6, 2, rng)
        assert sybil_n == 5
        assert replay_n == 2
        assert len(injected) == 7

    def test_observe_protocol_responses_feeds_correlation(self, rng):
        config = AttackConfig(active_attacks=[AttackType.CORRELATION])
        orch = AttackOrchestrator(config=config)
        responses = [
            PerturbedResponse(
                query_id=0,
                reported_x=100.0,
                reported_y=200.0,
                anomaly_confirmed=True,
                is_dummy=False,
            )
        ]
        orch.observe_protocol_responses(5, responses, time_window_steps=12)
        assert len(orch.correlation.trajectory_observations) == 1

    def test_evaluate_periodic_increments_when_distinguishable(self):
        config = AttackConfig(
            correlation_eval_interval=10,
            correlation_distinguish_threshold_m=50.0,
            active_attacks=[AttackType.CORRELATION],
        )
        orch = AttackOrchestrator(config=config)
        orch.correlation.trajectory_observations = [(1, 100.0, 100.0)]
        agent_x = np.array([100.0, 500.0, 600.0], dtype=np.float32)
        agent_y = np.array([100.0, 500.0, 600.0], dtype=np.float32)
        orch.evaluate_periodic(10, agent_x, agent_y)
        assert orch.correlation_evaluations == 1
        assert orch.correlation_successes == 1
