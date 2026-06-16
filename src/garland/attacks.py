"""Attack layer for GARLAND privacy protocol evaluation.

Implements adversarial strategies to test system robustness:
- Sybil injection (false positive flooding)
- Deanonymization via targeted queries
- Correlation attacks (temporal/spatial linking)
- Eclipse attacks (isolating zones)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from garland.privacy import (
    AnomalyType,
    BroadcastQuery,
    EncryptedToken,
    PerturbedResponse,
)


class AttackType(Enum):
    """Categories of network attacks."""

    SYBIL_INJECTION = "sybil_injection"
    TARGETED_QUERY = "targeted_query"
    CORRELATION = "correlation"
    ECLIPSE = "eclipse"
    REPLAY = "replay"


@dataclass
class AttackConfig:
    """Configuration for attack simulations.

    Parameters
    ----------
    sybil_count : int
        Number of fake identities per Sybil attack.
    sybil_target_zone : int
        Zone to flood with false anomaly reports.
    target_agent_idx : int
        Agent index the adversary is trying to deanonymize.
    correlation_window : int
        Steps to collect data for correlation attack.
    active_attacks : list[AttackType]
        Which attacks are currently active.
    """

    sybil_count: int = 20
    sybil_target_zone: int = 0
    target_agent_idx: int = 0
    correlation_window: int = 288  # 24 hours
    active_attacks: list[AttackType] = field(default_factory=list)


@dataclass
class SybilAttacker:
    """Sybil attack: inject many fake identities reporting anomalies.

    Goal: trigger false positive alerts in a target zone to either:
    1. Waste resources (denial of service on health response)
    2. Create cover noise to mask a real attack elsewhere
    """

    config: AttackConfig = field(default_factory=AttackConfig)
    injected_tokens: list[EncryptedToken] = field(default_factory=list)

    def generate_fake_tokens(
        self, target_zone: int, time_bin: int, count: int, rng: np.random.Generator
    ) -> list[EncryptedToken]:
        """Generate fake encrypted tokens from Sybil identities."""
        tokens = []
        for _ in range(count):
            token = EncryptedToken(
                zone_id=target_zone,
                anomaly_type=rng.choice(
                    [AnomalyType.RESPIRATORY, AnomalyType.FEBRILE]
                ),
                timestamp_bin=time_bin,
                agent_id_hash=int(rng.integers(0, 2**31)),
                is_dummy=False,
            )
            tokens.append(token)
        self.injected_tokens.extend(tokens)
        return tokens


@dataclass
class DeanonymizationAttacker:
    """Targeted query attack to unmask an individual agent's location.

    Strategy: Issue many narrow queries targeting a single cell,
    then correlate responses to isolate a specific individual.
    The K-anonymity spatial dilution should prevent this.
    """

    config: AttackConfig = field(default_factory=AttackConfig)
    observed_responses: list[PerturbedResponse] = field(default_factory=list)
    location_estimates: list[tuple[float, float]] = field(default_factory=list)

    def craft_targeted_query(
        self, target_cell: int, time_start: int, time_end: int, query_id: int
    ) -> BroadcastQuery:
        """Craft a query designed to isolate a specific agent."""
        return BroadcastQuery(
            zone_cells=[target_cell],  # Narrow single-cell target
            anomaly_type=AnomalyType.MULTI_SYSTEM,
            time_window_start=time_start,
            time_window_end=time_end,
            query_id=query_id,
        )

    def collect_response(self, response: PerturbedResponse) -> None:
        """Collect a response for correlation analysis."""
        self.observed_responses.append(response)
        if response.anomaly_confirmed and not response.is_dummy:
            self.location_estimates.append(
                (response.reported_x, response.reported_y)
            )

    def estimate_location(self) -> tuple[float, float] | None:
        """Attempt to estimate target agent's true location from responses.

        Uses centroid of collected perturbed locations as MLE estimate.
        """
        if not self.location_estimates:
            return None
        xs = [loc[0] for loc in self.location_estimates]
        ys = [loc[1] for loc in self.location_estimates]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def estimation_error(self, true_x: float, true_y: float) -> float | None:
        """Compute distance between estimated and true location."""
        estimate = self.estimate_location()
        if estimate is None:
            return None
        dx = estimate[0] - true_x
        dy = estimate[1] - true_y
        return float(np.sqrt(dx * dx + dy * dy))


@dataclass
class CorrelationAttacker:
    """Temporal/spatial correlation attack.

    Attempts to link responses across multiple queries to build
    a movement profile of a target agent.
    """

    config: AttackConfig = field(default_factory=AttackConfig)
    # (time_bin, reported_x, reported_y) tuples
    trajectory_observations: list[tuple[int, float, float]] = field(default_factory=list)

    def observe_response(
        self, time_bin: int, response: PerturbedResponse
    ) -> None:
        """Record a response observation for trajectory building."""
        if response.anomaly_confirmed:
            self.trajectory_observations.append(
                (time_bin, response.reported_x, response.reported_y)
            )

    def estimate_trajectory_length(self) -> float:
        """Estimate total movement distance from observations."""
        if len(self.trajectory_observations) < 2:
            return 0.0
        total = 0.0
        sorted_obs = sorted(self.trajectory_observations, key=lambda x: x[0])
        for i in range(1, len(sorted_obs)):
            dx = sorted_obs[i][1] - sorted_obs[i - 1][1]
            dy = sorted_obs[i][2] - sorted_obs[i - 1][2]
            total += np.sqrt(dx * dx + dy * dy)
        return total

    def can_distinguish_agents(
        self, agent_locations: NDArray[np.float32], threshold: float = 100.0
    ) -> bool:
        """Test if attacker can distinguish target from other agents.

        Returns True if the estimated position uniquely identifies one
        agent within threshold distance (attack success).
        Returns False if K-anonymity holds (attack failure).
        """
        if len(self.trajectory_observations) == 0:
            return False
        # Average observed position
        xs = [obs[1] for obs in self.trajectory_observations]
        ys = [obs[2] for obs in self.trajectory_observations]
        est_x, est_y = np.mean(xs), np.mean(ys)

        # Count agents within threshold of estimated position
        dx = agent_locations[:, 0] - est_x
        dy = agent_locations[:, 1] - est_y
        distances = np.sqrt(dx * dx + dy * dy)
        agents_in_range = np.sum(distances < threshold)

        # Attack succeeds only if exactly 1 agent is within threshold
        return int(agents_in_range) == 1


@dataclass
class EclipseAttacker:
    """Eclipse attack: surround target zone with controlled nodes.

    Goal: Control all relay/aggregation paths to a zone, allowing
    the attacker to selectively drop or modify messages.
    """

    config: AttackConfig = field(default_factory=AttackConfig)
    blocked_tokens: list[EncryptedToken] = field(default_factory=list)
    dropped_count: int = 0

    def intercept_token(
        self, token: EncryptedToken, target_zones: set[int]
    ) -> EncryptedToken | None:
        """Intercept and optionally drop tokens from target zones.

        Returns None if token is dropped (eclipsed), otherwise passes through.
        """
        if token.zone_id in target_zones:
            self.blocked_tokens.append(token)
            self.dropped_count += 1
            return None
        return token


@dataclass
class AttackOrchestrator:
    """Coordinates multiple attack types during simulation.

    Tracks attack success metrics for evaluation.
    """

    config: AttackConfig = field(default_factory=AttackConfig)
    sybil: SybilAttacker = field(default_factory=SybilAttacker)
    deanon: DeanonymizationAttacker = field(default_factory=DeanonymizationAttacker)
    correlation: CorrelationAttacker = field(default_factory=CorrelationAttacker)
    eclipse: EclipseAttacker = field(default_factory=EclipseAttacker)
    # Metrics
    false_positives_triggered: int = 0
    deanon_attempts: int = 0
    deanon_successes: int = 0

    def step(
        self,
        current_step: int,
        time_bin: int,
        rng: np.random.Generator,
    ) -> list[EncryptedToken]:
        """Execute active attacks for this step.

        Returns fake tokens to inject into the system.
        """
        fake_tokens: list[EncryptedToken] = []

        if AttackType.SYBIL_INJECTION in self.config.active_attacks:
            # Inject Sybil tokens periodically
            if current_step % 6 == 0:  # Every 30 minutes
                tokens = self.sybil.generate_fake_tokens(
                    target_zone=self.config.sybil_target_zone,
                    time_bin=time_bin,
                    count=self.config.sybil_count,
                    rng=rng,
                )
                fake_tokens.extend(tokens)

        return fake_tokens

    def evaluate_deanonymization(
        self, true_x: float, true_y: float, success_threshold: float = 50.0
    ) -> bool:
        """Check if deanonymization attack succeeded.

        Success = estimated location within success_threshold meters of true.
        """
        error = self.deanon.estimation_error(true_x, true_y)
        self.deanon_attempts += 1
        if error is not None and error < success_threshold:
            self.deanon_successes += 1
            return True
        return False
