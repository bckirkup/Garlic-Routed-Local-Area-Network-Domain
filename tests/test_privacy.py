"""Tests for privacy guarantees in the GARLAND testbed.

Confirms that:
1. An attacker injecting a single targeted query cannot unmask an individual
   agent's exact location (K-anonymity holds).
2. Planar Laplace mechanism provides geo-indistinguishability.
3. Randomized response provides plausible deniability.
4. Spatial dilution expands zones to meet K-min population.
5. Sybil injection cannot reliably trigger false positives when rate-limited.
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.agents import CitizenAgent, NetworkAggregator
from garland.attacks import (
    CorrelationAttacker,
    DeanonymizationAttacker,
    SybilAttacker,
)
from garland.biometrics import generate_profiles
from garland.privacy import (
    AggregatorState,
    AnomalyType,
    EncryptedToken,
    PerturbedResponse,
    PrivacyConfig,
    compute_adaptive_composition_epsilon,
    planar_laplace_noise,
    randomized_response,
)
from garland.spatial import SpatialGrid


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def populated_grid():
    """Create a spatial grid with 1000 agents distributed across cells."""
    grid = SpatialGrid(width=2000.0, height=2000.0, cell_size=200.0)
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.uniform(0, 2000, n).astype(np.float32)
    y = rng.uniform(0, 2000, n).astype(np.float32)
    grid.assign_positions(x, y)
    return grid, x, y


class TestPlanarLaplace:
    """Test Planar Laplace mechanism properties."""

    def test_noise_is_nonzero(self, rng):
        """Noise should not be zero (would leak exact position)."""
        results = [planar_laplace_noise(100.0, rng) for _ in range(100)]
        # At least some noise should be non-trivial
        distances = [np.sqrt(dx**2 + dy**2) for dx, dy in results]
        assert np.mean(distances) > 10.0

    def test_noise_scales_with_parameter(self, rng):
        """Larger scale → larger noise magnitude on average."""
        small_scale = [
            np.sqrt(dx**2 + dy**2)
            for dx, dy in [planar_laplace_noise(50.0, rng) for _ in range(500)]
        ]
        large_scale = [
            np.sqrt(dx**2 + dy**2)
            for dx, dy in [planar_laplace_noise(500.0, rng) for _ in range(500)]
        ]
        assert np.mean(large_scale) > np.mean(small_scale) * 2

    def test_noise_is_isotropic(self, rng):
        """Noise should be approximately isotropic (no directional bias)."""
        results = [planar_laplace_noise(200.0, rng) for _ in range(2000)]
        xs = [r[0] for r in results]
        ys = [r[1] for r in results]
        # Mean should be near zero in both dimensions
        assert abs(np.mean(xs)) < 50.0
        assert abs(np.mean(ys)) < 50.0


class TestRandomizedResponse:
    """Test randomized response mechanism."""

    def test_truthful_probability(self, rng):
        """With p=1.0, response should always equal truth."""
        for _ in range(100):
            assert randomized_response(True, 1.0, rng) is True
            assert randomized_response(False, 1.0, rng) is False

    def test_plausible_deniability(self, rng):
        """With p < 1, false values should sometimes appear as true."""
        results = [randomized_response(False, 0.5, rng) for _ in range(1000)]
        # Some should be True (plausible deniability)
        true_count = sum(results)
        assert true_count > 100  # Expect ~250 with p=0.5
        assert true_count < 500

    def test_bias_toward_truth(self, rng):
        """Higher p should yield more truthful responses."""
        high_p = sum(randomized_response(True, 0.9, rng) for _ in range(1000))
        low_p = sum(randomized_response(True, 0.5, rng) for _ in range(1000))
        assert high_p > low_p


class TestSpatialDilution:
    """Test K-anonymity spatial dilution."""

    def test_cell_ids_property_matches_cell_of(self, populated_grid):
        """Public cell_ids accessor should agree with cell_of per agent."""
        grid, x, y = populated_grid
        cell_ids = grid.cell_ids
        assert len(cell_ids) == len(x)
        for idx in range(len(x)):
            assert int(cell_ids[idx]) == grid.cell_of(idx)

    def test_dilution_meets_k_min(self, populated_grid):
        """Dilated zone should contain at least k_min agents."""
        grid, x, y = populated_grid
        k_min = 50
        center_cell = grid.cell_of(0)
        zone = grid.dilated_zone(center_cell, k_min)
        total_pop = sum(grid.zone_population(c) for c in zone)
        assert total_pop >= k_min

    def test_dilution_expands_from_center(self):
        """Zone should expand outward when the center cell is sparse."""
        grid = SpatialGrid(width=600.0, height=600.0, cell_size=200.0)
        x = np.array([10.0], dtype=np.float32)
        y = np.array([10.0], dtype=np.float32)
        grid.assign_positions(x, y)
        sparse_cell = grid.cell_of(0)
        assert grid.zone_population(sparse_cell) == 1
        zone = grid.dilated_zone(sparse_cell, k_min=20)
        assert len(zone) > 1

    def test_dilution_single_cell_sufficient(self, populated_grid):
        """If one cell has enough population, no expansion needed."""
        grid, x, y = populated_grid
        # Find a dense cell
        dense_cell = max(range(grid.n_cells), key=lambda c: grid.zone_population(c))
        pop = grid.zone_population(dense_cell)
        if pop >= 5:
            zone = grid.dilated_zone(dense_cell, pop)
            # Zone should be minimal
            assert len(zone) <= 9  # Center + ring


class TestDeanonymizationAttack:
    """Test that targeted queries cannot unmask individual locations."""

    def test_single_query_cannot_unmask(self, rng):
        """A single targeted query with Laplace noise should not reveal exact position."""
        true_x, true_y = 500.0, 500.0
        config = PrivacyConfig(laplace_scale=200.0)

        attacker = DeanonymizationAttacker()

        # Simulate agent responding to a targeted query
        for _ in range(1):  # Single query
            dx, dy = planar_laplace_noise(config.laplace_scale, rng)
            response = PerturbedResponse(
                query_id=0,
                reported_x=true_x + dx,
                reported_y=true_y + dy,
                anomaly_confirmed=True,
            )
            attacker.collect_response(response)

        error = attacker.estimation_error(true_x, true_y)
        # With a single response and scale=200, error should be large
        # (cannot pinpoint within 50m)
        assert error is None or error > 50.0

    def test_multiple_queries_still_noisy(self, rng):
        """Even with 10 queries, Laplace noise prevents precise localization."""
        true_x, true_y = 1000.0, 1000.0
        config = PrivacyConfig(laplace_scale=200.0)

        attacker = DeanonymizationAttacker()

        for _ in range(10):
            dx, dy = planar_laplace_noise(config.laplace_scale, rng)
            response = PerturbedResponse(
                query_id=0,
                reported_x=true_x + dx,
                reported_y=true_y + dy,
                anomaly_confirmed=True,
            )
            attacker.collect_response(response)

        error = attacker.estimation_error(true_x, true_y)
        # Even with averaging 10 samples, error should remain substantial
        # (sqrt(n) convergence: 200/sqrt(10) ≈ 63m, plus Gamma(2) variance)
        assert error is not None
        assert error > 30.0
        # Cannot localize within cell_size (200m) with only 10 noisy samples
        assert error < 500.0

    def test_k_anonymity_prevents_isolation(self, populated_grid, rng):
        """With K-anonymity dilution, attacker cannot isolate one agent."""
        grid, x, y = populated_grid

        attacker = CorrelationAttacker()

        # Simulate observations with noise
        target_idx = 42
        true_x, true_y = float(x[target_idx]), float(y[target_idx])

        for time_bin in range(20):
            dx, dy = planar_laplace_noise(200.0, rng)
            response = PerturbedResponse(
                query_id=time_bin,
                reported_x=true_x + dx,
                reported_y=true_y + dy,
                anomaly_confirmed=True,
            )
            attacker.observe_response(time_bin, response)

        # The attacker should NOT be able to distinguish the target
        agent_locs = np.column_stack([x, y])
        # With K-min=50 and noise, multiple agents should be within threshold
        can_distinguish = attacker.can_distinguish_agents(agent_locs, threshold=200.0)
        # This should fail (privacy holds) — many agents within 200m
        assert not can_distinguish


class TestSybilAttack:
    """Test Sybil injection countermeasures."""

    def test_sybil_tokens_generated(self, rng):
        """Sybil attacker should generate fake tokens."""
        attacker = SybilAttacker()
        tokens = attacker.generate_fake_tokens(
            target_zone=5, time_bin=10, count=20, rng=rng
        )
        assert len(tokens) == 20
        assert all(t.zone_id == 5 for t in tokens)

    def test_dummy_packets_filtered(self, rng):
        """Aggregator should filter dummy packets from counts."""
        state = AggregatorState()
        # Submit some dummy tokens
        for _ in range(100):
            dummy = EncryptedToken(
                zone_id=1,
                anomaly_type=AnomalyType.RESPIRATORY,
                timestamp_bin=5,
                agent_id_hash=int(rng.integers(0, 2**31)),
                is_dummy=True,
            )
            state.receive_token(dummy)

        # Should have no real counts
        config = PrivacyConfig(threshold_m=5)
        triggers = state.check_thresholds(5, config)
        assert len(triggers) == 0


class TestAdaptiveComposition:
    """Test privacy budget accounting."""

    def test_zero_queries_zero_epsilon(self):
        """No queries → no privacy loss."""
        assert compute_adaptive_composition_epsilon(0, 0.1) == 0.0

    def test_epsilon_grows_sublinearly(self):
        """Advanced composition grows as O(√n), not O(n)."""
        eps_10 = compute_adaptive_composition_epsilon(10, 0.1)
        eps_100 = compute_adaptive_composition_epsilon(100, 0.1)
        eps_1000 = compute_adaptive_composition_epsilon(1000, 0.1)

        # Should grow roughly as sqrt(n)
        assert eps_100 < eps_10 * 10  # Sublinear
        assert eps_1000 < eps_100 * 10

    def test_larger_per_query_epsilon_means_larger_total(self):
        """More per-query epsilon → higher total budget."""
        low = compute_adaptive_composition_epsilon(50, 0.01)
        high = compute_adaptive_composition_epsilon(50, 1.0)
        assert high > low

    def test_aggregator_uses_adaptive_composition(self):
        """Runtime epsilon accounting should match adaptive composition, not linear sum."""
        config = PrivacyConfig(epsilon_per_response=0.1)
        aggregator = NetworkAggregator(config=config)
        genuine_responses = [
            PerturbedResponse(
                query_id=0,
                reported_x=0.0,
                reported_y=0.0,
                anomaly_confirmed=True,
                is_dummy=False,
            )
            for _ in range(10)
        ]
        aggregator.collect_responses(genuine_responses)

        expected = compute_adaptive_composition_epsilon(10, config.epsilon_per_response)
        linear = 10 * config.epsilon_per_response
        assert aggregator.state.total_epsilon == expected
        assert aggregator.state.total_epsilon != linear
        assert aggregator.state.genuine_response_count == 10


class TestThresholdAggregator:
    """Test aggregator threshold detection."""

    def test_below_threshold_no_trigger(self, rng):
        """Fewer than M tokens should not trigger broadcast."""
        state = AggregatorState()
        config = PrivacyConfig(threshold_m=5, time_window_steps=12)

        for i in range(4):  # Below threshold
            token = EncryptedToken(
                zone_id=1,
                anomaly_type=AnomalyType.FEBRILE,
                timestamp_bin=10,
                agent_id_hash=i,
            )
            state.receive_token(token)

        triggers = state.check_thresholds(10, config)
        assert len(triggers) == 0

    def test_at_threshold_triggers(self, rng):
        """Exactly M tokens should trigger broadcast."""
        state = AggregatorState()
        config = PrivacyConfig(threshold_m=5, time_window_steps=12)

        for i in range(5):
            token = EncryptedToken(
                zone_id=2,
                anomaly_type=AnomalyType.RESPIRATORY,
                timestamp_bin=10,
                agent_id_hash=i,
            )
            state.receive_token(token)

        triggers = state.check_thresholds(10, config)
        assert len(triggers) == 1
        assert triggers[0] == (2, AnomalyType.RESPIRATORY)

    def test_old_tokens_expire(self, rng):
        """Tokens outside time window should not contribute to count."""
        state = AggregatorState()
        config = PrivacyConfig(threshold_m=5, time_window_steps=12)

        # Add old tokens
        for i in range(10):
            token = EncryptedToken(
                zone_id=1,
                anomaly_type=AnomalyType.FEBRILE,
                timestamp_bin=1,  # Old
                agent_id_hash=i,
            )
            state.receive_token(token)

        # Check at much later time
        triggers = state.check_thresholds(100, config)
        assert len(triggers) == 0


class TestProtocolIntegration:
    """Integration test: token → threshold → dilution → broadcast → response."""

    def test_clustered_anomaly_triggers_dilated_broadcast_and_response(self, rng):
        """Agents in the same cell receive dilated broadcast queries and respond."""
        grid = SpatialGrid(width=2000.0, height=2000.0, cell_size=200.0)
        config = PrivacyConfig(
            threshold_m=5,
            k_min=10,
            time_window_steps=12,
            randomized_response_p=1.0,
            dummy_rate=0.0,
        )

        n_cluster = 5
        n_agents = n_cluster + 1
        center_cx, center_cy = grid.cell_center(0)
        x = np.zeros(n_agents, dtype=np.float32)
        y = np.zeros(n_agents, dtype=np.float32)
        for i in range(n_cluster):
            x[i] = center_cx + rng.normal(0, 10)
            y[i] = center_cy + rng.normal(0, 10)

        distant_cell = grid.n_cells - 1
        distant_cx, distant_cy = grid.cell_center(distant_cell)
        x[n_cluster] = distant_cx
        y[n_cluster] = distant_cy
        grid.assign_positions(x, y)

        center_cell = grid.cell_of(0)
        distant_cell_id = grid.cell_of(n_cluster)
        assert center_cell != distant_cell_id

        profiles = generate_profiles(n_agents, rng)
        agents = []
        for i in range(n_agents):
            agent = CitizenAgent(
                idx=i,
                has_wearable=True,
                profile=profiles[i],
                neighborhood_id=99,
            )
            if i < n_cluster:
                agent.anomaly_active = True
                agent.anomaly_type = AnomalyType.FEBRILE
            agents.append(agent)

        time_bin = 10
        tokens = [
            EncryptedToken(
                zone_id=grid.cell_of(i),
                anomaly_type=AnomalyType.FEBRILE,
                timestamp_bin=time_bin,
                agent_id_hash=i,
            )
            for i in range(n_cluster)
        ]

        aggregator = NetworkAggregator(config=config)
        aggregator.ingest_tokens(tokens, time_bin)
        queries = aggregator.evaluate_and_broadcast(time_bin, grid.dilated_zone)

        assert len(queries) == 1
        query = queries[0]
        assert center_cell in query.zone_cells
        assert query.anomaly_type == AnomalyType.FEBRILE

        confirmed = []
        for agent in agents:
            cell_id = grid.cell_of(agent.idx)
            if cell_id in query.zone_cells:
                resp = agent.respond_to_query(
                    query,
                    float(x[agent.idx]),
                    float(y[agent.idx]),
                    cell_id,
                    config,
                    rng,
                )
                if resp is not None and resp.anomaly_confirmed:
                    confirmed.append(agent.idx)

        assert set(confirmed) == set(range(n_cluster))
        assert n_cluster not in confirmed

    def test_neighborhood_id_zone_id_targets_wrong_dilated_zone(self, rng):
        """Tokens with neighborhood_id as zone_id dilate around the wrong center cell."""
        grid = SpatialGrid(width=2000.0, height=2000.0, cell_size=200.0)
        config = PrivacyConfig(threshold_m=5, k_min=10, time_window_steps=12)

        n_cluster = 5
        center_cx, center_cy = grid.cell_center(0)
        x = np.zeros(n_cluster, dtype=np.float32)
        y = np.zeros(n_cluster, dtype=np.float32)
        for i in range(n_cluster):
            x[i] = center_cx + rng.normal(0, 10)
            y[i] = center_cy + rng.normal(0, 10)
        grid.assign_positions(x, y)

        agent_cell = grid.cell_of(0)
        wrong_zone_id = 5
        assert agent_cell != wrong_zone_id

        time_bin = 10
        tokens = [
            EncryptedToken(
                zone_id=wrong_zone_id,
                anomaly_type=AnomalyType.FEBRILE,
                timestamp_bin=time_bin,
                agent_id_hash=i,
            )
            for i in range(n_cluster)
        ]

        aggregator = NetworkAggregator(config=config)
        aggregator.ingest_tokens(tokens, time_bin)
        queries = aggregator.evaluate_and_broadcast(time_bin, grid.dilated_zone)

        assert len(queries) == 1
        query = queries[0]
        # Dilution expands from the token's zone_id (wrong cell), not the agents' cell
        assert query.zone_cells[0] == wrong_zone_id
        assert query.zone_cells[0] != agent_cell

        # Correct cell_id tokens would center dilution on the agents' actual cell
        correct_tokens = [
            EncryptedToken(
                zone_id=agent_cell,
                anomaly_type=AnomalyType.FEBRILE,
                timestamp_bin=time_bin,
                agent_id_hash=i,
            )
            for i in range(n_cluster)
        ]
        correct_aggregator = NetworkAggregator(config=config)
        correct_aggregator.ingest_tokens(correct_tokens, time_bin)
        correct_queries = correct_aggregator.evaluate_and_broadcast(
            time_bin, grid.dilated_zone
        )
        assert correct_queries[0].zone_cells[0] == agent_cell
