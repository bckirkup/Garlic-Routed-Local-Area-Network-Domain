"""Attack layer for GARLAND privacy protocol evaluation.

Implements adversarial strategies to test system robustness:
- Sybil injection (false positive flooding)
- Deanonymization via targeted queries
- Correlation attacks (temporal/spatial linking)
- Eclipse attacks (isolating zones)
- Replay attacks (stale token re-injection)
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
        Grid cell ID to flood with false anomaly reports.
    target_agent_idx : int
        Agent index the adversary is trying to deanonymize.
    correlation_window : int
        Steps of observation history for correlation attack.
    eclipse_target_zones : list[int]
        Grid cell IDs whose tokens are intercepted.
    eclipse_drop_fraction : float
        Fraction of tokens dropped in eclipsed zones.
    replay_interval_steps : int
        How often to inject replayed tokens.
    replay_lag_bins : int
        Minimum age (time bins) of cached tokens to replay.
    replay_count : int
        Number of stale tokens to inject per replay event.
    replay_cache_max : int
        Maximum cached tokens retained for replay.
    deanon_interval_steps : int
        Steps between deanonymization attempts.
    deanon_success_threshold_m : float
        Distance (meters) below which deanon counts as success.
    correlation_eval_interval : int
        Steps between correlation success evaluations.
    correlation_distinguish_threshold_m : float
        Distance threshold for correlation isolation test.
    active_attacks : list[AttackType]
        Which attacks are currently active.
    """

    sybil_count: int = 20
    sybil_target_zone: int = 0
    target_agent_idx: int = 0
    correlation_window: int = 288
    eclipse_target_zones: list[int] = field(default_factory=list)
    eclipse_drop_fraction: float = 1.0
    replay_interval_steps: int = 72
    replay_lag_bins: int = 3
    replay_count: int = 10
    replay_cache_max: int = 500
    deanon_interval_steps: int = 288
    deanon_success_threshold_m: float = 50.0
    correlation_eval_interval: int = 288
    correlation_distinguish_threshold_m: float = 100.0
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
        _anomaly_types = (AnomalyType.RESPIRATORY, AnomalyType.FEBRILE)
        for _ in range(count):
            token = EncryptedToken(
                zone_id=target_zone,
                anomaly_type=_anomaly_types[int(rng.integers(0, len(_anomaly_types)))],
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
            zone_cells=[target_cell],
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
    trajectory_observations: list[tuple[int, float, float]] = field(default_factory=list)

    def observe_response(
        self, time_bin: int, response: PerturbedResponse
    ) -> None:
        """Record a response observation for trajectory building."""
        if response.anomaly_confirmed:
            self.trajectory_observations.append(
                (time_bin, response.reported_x, response.reported_y)
            )

    def prune_observations(self, time_bin: int, time_window_steps: int) -> None:
        """Drop observations outside the configured correlation window."""
        window_bins = max(1, self.config.correlation_window // time_window_steps)
        cutoff = time_bin - window_bins
        self.trajectory_observations = [
            obs for obs in self.trajectory_observations if obs[0] >= cutoff
        ]

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
        xs = [obs[1] for obs in self.trajectory_observations]
        ys = [obs[2] for obs in self.trajectory_observations]
        est_x, est_y = np.mean(xs), np.mean(ys)

        dx = agent_locations[:, 0] - est_x
        dy = agent_locations[:, 1] - est_y
        distances = np.sqrt(dx * dx + dy * dy)
        agents_in_range = np.sum(distances < threshold)

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
        self,
        token: EncryptedToken,
        target_zones: set[int],
        rng: np.random.Generator,
        drop_fraction: float = 1.0,
    ) -> EncryptedToken | None:
        """Intercept and optionally drop tokens from target zones.

        Returns None if token is dropped (eclipsed), otherwise passes through.
        """
        if token.zone_id not in target_zones:
            return token
        if drop_fraction < 1.0 and rng.random() > drop_fraction:
            return token
        self.blocked_tokens.append(token)
        self.dropped_count += 1
        return None


@dataclass
class ReplayAttacker:
    """Replay attack: re-inject stale anomaly tokens to trigger spurious alerts."""

    config: AttackConfig = field(default_factory=AttackConfig)
    token_cache: list[EncryptedToken] = field(default_factory=list)
    replayed_tokens: list[EncryptedToken] = field(default_factory=list)
    replay_inject_count: int = 0
    last_replay_zones: set[int] = field(default_factory=set)

    def cache_tokens(self, tokens: list[EncryptedToken]) -> None:
        """Cache non-dummy tokens for potential replay."""
        for token in tokens:
            if not token.is_dummy:
                self.token_cache.append(token)
        if len(self.token_cache) > self.config.replay_cache_max:
            self.token_cache = self.token_cache[-self.config.replay_cache_max :]

    def generate_replay_tokens(
        self, current_step: int, time_bin: int, rng: np.random.Generator
    ) -> list[EncryptedToken]:
        """Re-submit aged tokens with the current time bin."""
        self.last_replay_zones = set()
        if AttackType.REPLAY not in self.config.active_attacks:
            return []
        if current_step % self.config.replay_interval_steps != 0:
            return []
        if not self.token_cache:
            return []

        lag = self.config.replay_lag_bins
        candidates = [
            token
            for token in self.token_cache
            if token.timestamp_bin <= time_bin - lag
        ]
        if not candidates:
            candidates = self.token_cache[: min(10, len(self.token_cache))]

        count = min(self.config.replay_count, len(candidates))
        if count == 0:
            return []

        indices = rng.choice(len(candidates), size=count, replace=False)
        replayed: list[EncryptedToken] = []
        for idx in indices:
            old = candidates[int(idx)]
            new_token = EncryptedToken(
                zone_id=old.zone_id,
                anomaly_type=old.anomaly_type,
                timestamp_bin=time_bin,
                agent_id_hash=old.agent_id_hash,
                is_dummy=False,
            )
            replayed.append(new_token)
            self.last_replay_zones.add(old.zone_id)

        self.replayed_tokens.extend(replayed)
        self.replay_inject_count += len(replayed)
        return replayed


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
    replay: ReplayAttacker = field(default_factory=ReplayAttacker)
    false_positives_triggered: int = 0
    deanon_attempts: int = 0
    deanon_successes: int = 0
    correlation_evaluations: int = 0
    correlation_successes: int = 0
    replay_false_alerts: int = 0

    def __post_init__(self) -> None:
        self._sync_sub_configs()

    def _sync_sub_configs(self) -> None:
        """Propagate shared config to sub-attackers."""
        self.sybil.config = self.config
        self.deanon.config = self.config
        self.correlation.config = self.config
        self.eclipse.config = self.config
        self.replay.config = self.config

    def filter_tokens(
        self, tokens: list[EncryptedToken], rng: np.random.Generator
    ) -> tuple[list[EncryptedToken], int]:
        """Apply eclipse filtering before aggregation."""
        if AttackType.ECLIPSE not in self.config.active_attacks:
            return tokens, 0

        target_zones = set(self.config.eclipse_target_zones)
        if not target_zones:
            return tokens, 0

        before = self.eclipse.dropped_count
        filtered: list[EncryptedToken] = []
        for token in tokens:
            result = self.eclipse.intercept_token(
                token,
                target_zones,
                rng,
                self.config.eclipse_drop_fraction,
            )
            if result is not None:
                filtered.append(result)
        return filtered, self.eclipse.dropped_count - before

    def step_injections(
        self,
        current_step: int,
        time_bin: int,
        rng: np.random.Generator,
    ) -> tuple[list[EncryptedToken], int, int]:
        """Execute Sybil and replay injections for this step."""
        injected: list[EncryptedToken] = []
        sybil_count = 0
        replay_count = 0

        if AttackType.SYBIL_INJECTION in self.config.active_attacks:
            if current_step % 6 == 0:
                tokens = self.sybil.generate_fake_tokens(
                    target_zone=self.config.sybil_target_zone,
                    time_bin=time_bin,
                    count=self.config.sybil_count,
                    rng=rng,
                )
                injected.extend(tokens)
                sybil_count = len(tokens)

        if AttackType.REPLAY in self.config.active_attacks:
            replay_tokens = self.replay.generate_replay_tokens(
                current_step, time_bin, rng
            )
            injected.extend(replay_tokens)
            replay_count = len(replay_tokens)

        return injected, sybil_count, replay_count

    def cache_tokens_for_replay(self, tokens: list[EncryptedToken]) -> None:
        """Retain tokens for future replay injection."""
        if AttackType.REPLAY in self.config.active_attacks:
            self.replay.cache_tokens(tokens)

    def observe_protocol_responses(
        self,
        time_bin: int,
        responses: list[PerturbedResponse],
        time_window_steps: int,
    ) -> None:
        """Feed broadcast responses to the correlation attacker."""
        if AttackType.CORRELATION not in self.config.active_attacks:
            return

        for response in responses:
            if response.anomaly_confirmed and not response.is_dummy:
                self.correlation.observe_response(time_bin, response)
        self.correlation.prune_observations(time_bin, time_window_steps)

    def evaluate_periodic(
        self,
        current_step: int,
        agent_x: NDArray[np.float32],
        agent_y: NDArray[np.float32],
    ) -> None:
        """Score correlation attack success on a periodic schedule."""
        if AttackType.CORRELATION not in self.config.active_attacks:
            return
        if current_step % self.config.correlation_eval_interval != 0:
            return
        if not self.correlation.trajectory_observations:
            return

        self.correlation_evaluations += 1
        agent_locations = np.column_stack([agent_x, agent_y]).astype(np.float32)
        if self.correlation.can_distinguish_agents(
            agent_locations,
            self.config.correlation_distinguish_threshold_m,
        ):
            self.correlation_successes += 1

    def record_replay_false_alerts(self, query_zone_cells: list[int]) -> None:
        """Count broadcasts triggered in zones targeted by replay injection."""
        if not self.replay.last_replay_zones:
            return
        for zone in self.replay.last_replay_zones:
            if zone in query_zone_cells:
                self.replay_false_alerts += 1

    def evaluate_deanonymization(
        self, true_x: float, true_y: float, success_threshold: float = 50.0
    ) -> bool:
        """Check if deanonymization attack succeeded."""
        error = self.deanon.estimation_error(true_x, true_y)
        self.deanon_attempts += 1
        if error is not None and error < success_threshold:
            self.deanon_successes += 1
            return True
        return False

    def step(
        self,
        current_step: int,
        time_bin: int,
        rng: np.random.Generator,
    ) -> list[EncryptedToken]:
        """Backward-compatible injection hook (Sybil + replay only)."""
        injected, _, _ = self.step_injections(current_step, time_bin, rng)
        return injected
