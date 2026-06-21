"""Agent definitions for the GARLAND epidemiological security testbed.

Defines:
- CitizenAgent: The edge device (wearable BAN) with biometric monitoring
- MaliciousAgent: Sybil/adversarial agent for attack testing
- NetworkAggregator: Secure threshold aggregation node
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.biometrics import (
    BaselineTracker,
    BiometricProfile,
    generate_observation,
)
from garland.privacy import (
    AggregatorState,
    AnomalyType,
    BroadcastQuery,
    EncryptedToken,
    PerturbedResponse,
    PrivacyConfig,
    classify_anomaly,
    planar_laplace_noise,
    randomized_response,
)

# Anomaly detection threshold (Mahalanobis distance)
ANOMALY_THRESHOLD = 3.5


@dataclass
class CitizenAgent:
    """Edge device agent representing a wearable body area network.

    Only agents with `has_wearable=True` execute the full biometric pipeline.
    Others participate in spatial dynamics only (for SEIR transmission).

    Parameters
    ----------
    idx : int
        Unique agent index.
    has_wearable : bool
        Whether this agent has a wearable device.
    profile : BiometricProfile | None
        Biometric resting profile (None if no wearable).
    household_id : int
        Household cluster (wearable penetration is household-patchy).
    neighborhood_id : int
        Neighborhood cluster for population layout (not used by privacy protocol).
    """

    idx: int
    has_wearable: bool = False
    profile: BiometricProfile | None = None
    household_id: int = 0
    neighborhood_id: int = 0

    # State
    baseline: BaselineTracker = field(default_factory=BaselineTracker)
    anomaly_active: bool = False
    anomaly_type: AnomalyType | None = None
    last_observation: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    queries_answered: int = 0
    local_epsilon: float = 0.0

    def observe_and_detect(
        self,
        hour: int,
        month: int,
        day_of_year: int,
        hour_of_day: float,
        rng: np.random.Generator,
        cell_id: int,
        hazard_perturbation: NDArray[np.float64] | None = None,
        activity_level: float = 0.0,
    ) -> EncryptedToken | None:
        """Generate biometric observation, update baseline, detect anomalies.

        Returns an encrypted token if anomaly detected, else None.
        """
        if not self.has_wearable or self.profile is None:
            return None

        # Generate observation with any hazard effects
        obs = generate_observation(
            self.profile, hour_of_day, day_of_year, rng, activity_level
        )
        if hazard_perturbation is not None:
            obs += hazard_perturbation

        self.last_observation = obs

        # Compute anomaly score
        maha_dist = self.baseline.mahalanobis_distance(obs, hour, month)

        # Update baseline (adaptive forgetting)
        self.baseline.update(obs, hour, month)

        # Check anomaly predicate
        if maha_dist > ANOMALY_THRESHOLD:
            baseline_expected = self.baseline.expected_baseline(hour, month)
            atype = classify_anomaly(obs, baseline_expected)
            if atype is not None:
                self.anomaly_active = True
                self.anomaly_type = atype
                # Generate blind-gated encrypted token
                return EncryptedToken(
                    zone_id=cell_id,
                    anomaly_type=atype,
                    timestamp_bin=0,  # Set by caller
                    agent_id_hash=hash(self.idx) & 0x7FFFFFFF,
                )
        else:
            self.anomaly_active = False
            self.anomaly_type = None

        return None

    def respond_to_query(
        self,
        query: BroadcastQuery,
        true_x: float,
        true_y: float,
        cell_id: int,
        config: PrivacyConfig,
        rng: np.random.Generator,
    ) -> PerturbedResponse | None:
        """Evaluate and respond to a reverse-query broadcast.

        Applies:
        1. Randomized Response (coin-flip DP)
        2. Planar Laplace noise for geo-indistinguishability
        """
        if not self.has_wearable:
            return None

        # Check if agent matches the query criteria
        matches = (
            self.anomaly_active
            and self.anomaly_type == query.anomaly_type
            and cell_id in query.zone_cells
        )

        # Randomized response
        reported_match = randomized_response(matches, config.randomized_response_p, rng)

        if not reported_match:
            # Non-matching: optionally emit dummy packet
            if rng.random() < config.dummy_rate:
                dx, dy = planar_laplace_noise(config.laplace_scale * 2, rng)
                return PerturbedResponse(
                    query_id=query.query_id,
                    reported_x=true_x + dx,
                    reported_y=true_y + dy,
                    anomaly_confirmed=False,
                    is_dummy=True,
                )
            return None

        # Apply Planar Laplace noise to location
        dx, dy = planar_laplace_noise(config.laplace_scale, rng)
        perturbed_x = true_x + dx
        perturbed_y = true_y + dy

        # Track privacy budget
        self.queries_answered += 1
        self.local_epsilon += config.epsilon_per_response

        return PerturbedResponse(
            query_id=query.query_id,
            reported_x=perturbed_x,
            reported_y=perturbed_y,
            anomaly_confirmed=True,
            is_dummy=False,
        )

    def generate_dummy_traffic(
        self,
        true_x: float,
        true_y: float,
        cell_id: int,
        config: PrivacyConfig,
        rng: np.random.Generator,
    ) -> EncryptedToken | None:
        """Periodically emit dummy noise packets for traffic obfuscation."""
        if not self.has_wearable:
            return None
        if rng.random() < config.dummy_rate:
            return EncryptedToken(
                zone_id=cell_id,
                anomaly_type=rng.choice(list(AnomalyType)),
                timestamp_bin=0,
                agent_id_hash=int(rng.integers(0, 2**31)),
                is_dummy=True,
            )
        return None


@dataclass
class MaliciousAgent:
    """Adversarial agent for testing Sybil and deanonymization attacks.

    Can:
    - Generate fake anomaly tokens (Sybil injection)
    - Issue crafted queries attempting to unmask individuals
    - Collect and correlate responses for trajectory building
    """

    idx: int
    target_zone: int = 0
    target_agent: int = 0
    sybil_identities: int = 20

    # Attack state
    injected_count: int = 0
    collected_responses: list[PerturbedResponse] = field(default_factory=list)
    estimated_positions: list[tuple[float, float]] = field(default_factory=list)

    def inject_sybil_tokens(
        self, time_bin: int, rng: np.random.Generator
    ) -> list[EncryptedToken]:
        """Generate fake tokens from synthetic identities."""
        tokens = []
        for _ in range(self.sybil_identities):
            token = EncryptedToken(
                zone_id=self.target_zone,
                anomaly_type=AnomalyType.RESPIRATORY,
                timestamp_bin=time_bin,
                agent_id_hash=int(rng.integers(0, 2**31)),
                is_dummy=False,
            )
            tokens.append(token)
            self.injected_count += 1
        return tokens

    def collect_response(self, response: PerturbedResponse) -> None:
        """Observe and store a response for analysis."""
        self.collected_responses.append(response)
        if response.anomaly_confirmed and not response.is_dummy:
            self.estimated_positions.append(
                (response.reported_x, response.reported_y)
            )

    def estimate_target_location(self) -> tuple[float, float] | None:
        """MLE estimate of target location from perturbed responses."""
        if not self.estimated_positions:
            return None
        xs = [p[0] for p in self.estimated_positions]
        ys = [p[1] for p in self.estimated_positions]
        return (float(np.mean(xs)), float(np.mean(ys)))


@dataclass
class NetworkAggregator:
    """Secure threshold aggregation node.

    Cannot decrypt individual tokens. Performs:
    1. Homomorphic token counting per zone
    2. Threshold detection
    3. K-anonymity spatial dilution
    4. Broadcast query generation
    """

    config: PrivacyConfig = field(default_factory=PrivacyConfig)
    state: AggregatorState = field(default_factory=AggregatorState)
    broadcasts_issued: int = 0
    total_responses_received: int = 0

    def ingest_tokens(self, tokens: list[EncryptedToken], time_bin: int) -> None:
        """Receive batch of encrypted tokens for aggregation."""
        for token in tokens:
            token_with_time = EncryptedToken(
                zone_id=token.zone_id,
                anomaly_type=token.anomaly_type,
                timestamp_bin=time_bin,
                agent_id_hash=token.agent_id_hash,
                is_dummy=token.is_dummy,
            )
            self.state.receive_token(token_with_time)

    def evaluate_and_broadcast(
        self,
        current_time_bin: int,
        spatial_dilate_fn,
    ) -> list[BroadcastQuery]:
        """Check thresholds and generate dilated broadcast queries."""
        triggers = self.state.check_thresholds(current_time_bin, self.config)
        queries = []

        for zone_id, anomaly_type in triggers:
            # Apply K-anonymity spatial dilution
            dilated_cells = spatial_dilate_fn(zone_id, self.config.k_min)

            query = BroadcastQuery(
                zone_cells=dilated_cells,
                anomaly_type=anomaly_type,
                time_window_start=current_time_bin - self.config.time_window_steps,
                time_window_end=current_time_bin,
                query_id=self.broadcasts_issued,
            )
            queries.append(query)
            self.broadcasts_issued += 1

        return queries

    def collect_responses(self, responses: list[PerturbedResponse]) -> None:
        """Collect perturbed responses from broadcast."""
        self.state.responses.extend(responses)
        self.total_responses_received += len(responses)
        # Record epsilon for each genuine response
        genuine = sum(1 for r in responses if r.anomaly_confirmed and not r.is_dummy)
        if genuine > 0:
            epsilon = genuine * self.config.epsilon_per_response
            self.state.record_epsilon(epsilon)
