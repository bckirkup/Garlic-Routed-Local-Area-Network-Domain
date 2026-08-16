"""Decentralized privacy and routing protocol simulation for GARLAND.

Implements the broadcast-and-filter privacy design:
- Blind Gating (plaintext token simulation)
- Secure Threshold Aggregator
- Spatial Dilution (K-Anonymity)
- Reverse-Query Broadcast
- Uplink Perturbation (Randomized Response + Planar Laplace)
- Traffic Obfuscation (dummy noise packets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from garland.channels import (
    DEFAULT_CHANNEL_SET,
    MULTI_SYSTEM_MIN_CHANNELS,
    ChannelSet,
    ChannelSystem,
)
from garland.disambiguation import DisambiguationHypothesis


class AnomalyType(Enum):
    """Types of biometric anomalies detected by agents."""

    RESPIRATORY = "respiratory"
    CARDIAC = "cardiac"
    FEBRILE = "febrile"
    MULTI_SYSTEM = "multi_system"


class EncryptedToken(NamedTuple):
    """Simulated token-shaped anomaly report.

    Production encryption is not implemented; this tuple only simulates the
    protocol data shape.
    """

    zone_id: int
    anomaly_type: AnomalyType
    timestamp_bin: int  # Quantized time window
    agent_id_hash: int  # Cryptographic hash (not reversible)
    is_dummy: bool = False  # Traffic obfuscation dummy


@dataclass
class PrivacyConfig:
    """Privacy protocol parameters.

    Parameters
    ----------
    threshold_m : int
        Minimum anomaly count to trigger a broadcast.
    k_min : int
        K-anonymity minimum population for spatial dilution.
    time_window_steps : int
        Aggregation window in 5-min steps (default: 12 = 1 hour).
    epsilon_per_response : float
        Legacy configured privacy cost per randomized response.
    randomized_response_p : float
        Probability of truthful response in coin-flip RR.
    response_epsilon_basis : {"mechanism", "legacy"}
        Accounting basis for randomized-response channels. The mechanism
        basis derives epsilon from ``randomized_response_p``; legacy retains
        the configured constant for reproduction of historical runs.
    geo_epsilon_basis : {"separate"}
        Reporting basis for the planar geo channel. It is reported separately
        and is not added to response-channel epsilon.
    laplace_scale : float
        Scale parameter for Planar Laplace mechanism (meters).
    dummy_rate : float
        Rate at which non-matching agents emit dummy packets.
    dilation_basis : {"residents", "observed_devices", "true_devices"}
        Population basis for spatial dilation. ``true_devices`` is an
        evaluation-only oracle reference and is never the default.
    dilation_window_steps : int
        Trailing traffic window used by the observed-device estimator. When
        omitted, it defaults to ``time_window_steps``.
    dilation_margin_factor : float
        Conservative Poisson-style lower-bound margin in standard deviations.
    """

    threshold_m: int = 5
    k_min: int = 50
    time_window_steps: int = 12
    epsilon_per_response: float = 0.1
    randomized_response_p: float = 0.75
    laplace_scale: float = 200.0
    response_epsilon_basis: Literal["mechanism", "legacy"] = "mechanism"
    geo_epsilon_basis: Literal["separate"] = "separate"
    dummy_rate: float = 0.01
    dilation_basis: Literal["residents", "observed_devices", "true_devices"] = "observed_devices"
    dilation_window_steps: int | None = None
    dilation_margin_factor: float = 0.5

    def __post_init__(self) -> None:
        """Validate the configured population basis and estimator parameters."""
        if self.dilation_window_steps is None:
            self.dilation_window_steps = self.time_window_steps
        if self.response_epsilon_basis not in {"mechanism", "legacy"}:
            raise ValueError("response_epsilon_basis must be 'mechanism' or 'legacy'")
        if self.geo_epsilon_basis != "separate":
            raise ValueError("geo_epsilon_basis must be 'separate'")
        if self.randomized_response_p < 0.0 or self.randomized_response_p > 1.0:
            raise ValueError("randomized_response_p must be between 0 and 1")
        if self.laplace_scale <= 0.0:
            raise ValueError("laplace_scale must be positive")
        if self.dilation_basis not in {"residents", "observed_devices", "true_devices"}:
            raise ValueError(
                "dilation_basis must be 'residents', 'observed_devices', or 'true_devices'"
            )
        if self.dilation_window_steps <= 0:
            raise ValueError("dilation_window_steps must be positive")
        if self.dilation_margin_factor < 0:
            raise ValueError("dilation_margin_factor must be non-negative")

    def response_epsilon(self) -> float:
        """Return the configured accounting cost for one RR response."""
        if self.response_epsilon_basis == "legacy":
            return self.epsilon_per_response
        return randomized_response_epsilon(self.randomized_response_p)

    def geo_epsilon(self) -> float:
        """Return planar-Laplace epsilon as a separately reported quantity."""
        return 1.0 / self.laplace_scale


@dataclass
class AggregatorState:
    """State of the Secure Threshold Aggregator.

    Cannot read individual tokens — only counts matching patterns.
    """

    # Zone → AnomalyType → list of timestamps
    token_counts: dict[int, dict[AnomalyType, list[int]]] = field(default_factory=dict)
    # Zone → timestamp bin → observed packet count, including dummies
    observed_traffic: dict[int, dict[int, int]] = field(default_factory=dict)
    # Zone → timestamp bin → genuine anomaly-token count
    anomaly_traffic: dict[int, dict[int, int]] = field(default_factory=dict)
    # Active broadcasts
    active_queries: list[BroadcastQuery] = field(default_factory=list)
    # Responses collected
    responses: list[PerturbedResponse] = field(default_factory=list)
    # Indicative adaptive-composition accounting over genuine responses
    total_epsilon: float = 0.0
    genuine_response_count: int = 0
    # History of epsilon expenditure per step
    epsilon_history: list[float] = field(default_factory=list)
    disambiguation_answer_count: int = 0
    disambiguation_ack_release_count: int = 0
    disambiguation_answer_epsilon: float = 0.0
    disambiguation_ack_epsilon: float = 0.0
    response_epsilon_per_response: float = 0.0

    def _update_total_epsilon(self, delta: float = 1e-6) -> None:
        """Recompute cumulative epsilon across all protocol channels."""
        self.total_epsilon = (
            compute_adaptive_composition_epsilon(
                self.genuine_response_count,
                self.response_epsilon_per_response,
                delta,
            )
            + self.disambiguation_answer_epsilon
            + self.disambiguation_ack_epsilon
        )
        self.epsilon_history.append(self.total_epsilon)

    def receive_token(self, token: EncryptedToken) -> None:
        """Ingest a token-shaped report into the aggregate simulation."""
        traffic_by_bin = self.observed_traffic.setdefault(token.zone_id, {})
        traffic_by_bin[token.timestamp_bin] = traffic_by_bin.get(token.timestamp_bin, 0) + 1
        if token.is_dummy:
            return  # Dummy packets are filtered by the aggregator
        zone = token.zone_id
        atype = token.anomaly_type
        if zone not in self.token_counts:
            self.token_counts[zone] = {}
        if atype not in self.token_counts[zone]:
            self.token_counts[zone][atype] = []
        self.token_counts[zone][atype].append(token.timestamp_bin)
        anomaly_by_bin = self.anomaly_traffic.setdefault(zone, {})
        anomaly_by_bin[token.timestamp_bin] = anomaly_by_bin.get(token.timestamp_bin, 0) + 1

    def check_thresholds(
        self, current_time_bin: int, config: PrivacyConfig
    ) -> list[tuple[int, AnomalyType]]:
        """Check if any zone × anomaly exceeds threshold within window.

        Returns list of (zone_id, anomaly_type) pairs that triggered.
        """
        triggers = []
        window_start = current_time_bin - config.time_window_steps

        for zone, atypes in self.token_counts.items():
            for atype, timestamps in atypes.items():
                # Count tokens within the current window
                recent = [t for t in timestamps if t >= window_start]
                if len(recent) >= config.threshold_m:
                    triggers.append((zone, atype))
                    # Clear processed tokens to avoid re-triggering
                    self.token_counts[zone][atype] = [
                        t for t in timestamps if t >= current_time_bin
                    ]
        return triggers

    def estimate_observed_devices(
        self,
        cell_id: int,
        current_time_bin: int,
        config: PrivacyConfig,
        current_step: int | None = None,
    ) -> int:
        """Estimate current device occupancy from recent observed traffic.

        A shorter trailing window makes the estimate more current but noisier.
        Devices moving between cells can therefore still cause venue-clustering
        lag under schedule mobility.
        """
        traffic_by_bin = self.observed_traffic.get(cell_id)
        if not traffic_by_bin or config.dummy_rate <= 0:
            return 0
        window_steps = config.dilation_window_steps or config.time_window_steps
        window_bins = max(1, int(np.ceil(window_steps / config.time_window_steps)))
        window_start = current_time_bin - window_bins + 1
        for stamp in tuple(traffic_by_bin):
            if stamp < window_start:
                del traffic_by_bin[stamp]
        anomaly_by_bin = self.anomaly_traffic.get(cell_id, {})
        for stamp in tuple(anomaly_by_bin):
            if stamp < window_start:
                del anomaly_by_bin[stamp]
        observed = sum(count for stamp, count in traffic_by_bin.items() if stamp >= window_start)
        anomaly_tokens = sum(
            count for stamp, count in anomaly_by_bin.items() if stamp >= window_start
        )
        observed = max(0, observed - anomaly_tokens)
        available_steps = (
            current_step + 1
            if current_step is not None
            else (current_time_bin + 1) * config.time_window_steps
        )
        history_steps = min(window_steps, max(1, available_steps))
        lower_bound = max(0.0, observed - config.dilation_margin_factor * np.sqrt(observed))
        return int(lower_bound / (config.dummy_rate * history_steps))

    def record_genuine_responses(
        self, count: int, epsilon_per_response: float, delta: float = 1e-6
    ) -> None:
        """Track the selected response-channel basis through composition."""
        if count <= 0:
            return
        self.response_epsilon_per_response = epsilon_per_response
        self.genuine_response_count += count
        self._update_total_epsilon(delta)

    def record_disambiguation_answers(
        self, count: int, epsilon_per_response: float, delta: float = 1e-6
    ) -> None:
        """Charge every approved yes/no answer through adaptive composition."""
        if count <= 0:
            return
        self.disambiguation_answer_count += count
        self.disambiguation_answer_epsilon = compute_adaptive_composition_epsilon(
            self.disambiguation_answer_count, epsilon_per_response, delta
        )
        self._update_total_epsilon(delta)

    def record_disambiguation_ack(self, epsilon: float, delta: float = 1e-6) -> None:
        """Charge one released zone-level acknowledgement count separately."""
        self.disambiguation_ack_release_count += 1
        self.disambiguation_ack_epsilon += epsilon
        self._update_total_epsilon(delta)


@dataclass
class BroadcastQuery:
    """Reverse-query broadcast sent to devices in a dilated zone."""

    zone_cells: list[int]  # Dilated zone (after K-anonymity expansion)
    anomaly_type: AnomalyType
    time_window_start: int
    time_window_end: int
    query_id: int = 0


@dataclass(frozen=True)
class DisambiguationQuery:
    """Human-approved hypothesis query over a previously dilated zone."""

    zone_cells: list[int]
    hypothesis: DisambiguationHypothesis
    referenced_query_ids: tuple[int, ...]
    time_window_start: int
    time_window_end: int
    query_id: int = 0


@dataclass
class PerturbedResponse:
    """Agent response with geo-indistinguishable location perturbation."""

    query_id: int
    reported_x: float  # Perturbed location
    reported_y: float  # Perturbed location
    anomaly_confirmed: bool
    is_dummy: bool = False


def noised_count_with_floor(
    count: int,
    population: int,
    k_min: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> int:
    """Release a bounded noisy count only when the population meets k-anonymity."""
    if population < k_min:
        return 0
    noise = int(round(float(rng.laplace(0.0, noise_scale))))
    return max(0, min(population, count + noise))


def planar_laplace_noise(scale: float, rng: np.random.Generator) -> tuple[float, float]:
    """Generate 2D Planar Laplace noise for a geo-privacy design.

    The scale corresponds to a declared geo epsilon of ``1 / scale``. A
    formal geo-indistinguishability proof is outside this simulation.

    Parameters
    ----------
    scale : float
        Scale parameter (1/ε in geo-indistinguishability terms).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    (dx, dy) : tuple[float, float]
        Noise to add to true coordinates.
    """
    # Sample from Gamma(2, 1/ε) for radius, uniform for angle
    # This samples the radial distribution used by the design.
    theta = rng.uniform(0, 2 * np.pi)
    # Inverse CDF method for Gamma(2, scale)
    r = -scale * (np.log(rng.random()) + np.log(rng.random()))
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return (float(dx), float(dy))


def randomized_response(true_value: bool, p: float, rng: np.random.Generator) -> bool:
    """Coin-flip randomized response mechanism.

    With probability p, report truthfully. Otherwise, flip a fair coin.
    The mechanism-derived per-response epsilon is
    ``ln((1+p)/(1-p))`` for ``0 <= p < 1``. This local mechanism does not by
    itself establish privacy for repeated correlated queries.

    Parameters
    ----------
    true_value : bool
        The agent's actual anomaly status.
    p : float
        Probability of truthful response.
    rng : np.random.Generator
        Random number generator.
    """
    if rng.random() < p:
        return true_value
    return rng.random() < 0.5


def randomized_response_epsilon(p: float) -> float:
    """Return the mechanism-derived epsilon for coin-flip randomized response."""
    if p < 0.0 or p > 1.0:
        raise ValueError("randomized-response truth probability must be between 0 and 1")
    if p == 1.0:
        return float("inf")
    if p == 0.0:
        return 0.0
    return float(np.log((1.0 + p) / (1.0 - p)))


def compute_adaptive_composition_epsilon(
    n_queries: int, epsilon_per_query: float, delta: float = 1e-6
) -> float:
    """Compute an indicative advanced-composition-style total.

    This calculation is not a proven bound for data-dependent broadcasts:
    ε_total = ε√(2n·ln(1/δ)) + n·ε·(e^ε - 1)

    For small ε, this approximates: ε_total ≈ ε√(2n·ln(1/δ))

    Parameters
    ----------
    n_queries : int
        Number of queries answered.
    epsilon_per_query : float
        Per-query epsilon.
    delta : float
        Failure probability.
    """
    if n_queries == 0:
        return 0.0
    eps = epsilon_per_query
    # Advanced composition theorem
    term1 = eps * np.sqrt(2 * n_queries * np.log(1.0 / delta))
    term2 = n_queries * eps * (np.exp(eps) - 1)
    return float(term1 + term2)


def _system_exceeded(
    diff: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    channel_set: ChannelSet,
    system: ChannelSystem,
    observed: NDArray[np.bool_],
    *,
    signed: bool = False,
) -> bool:
    """Whether any reported channel of ``system`` exceeds its threshold."""
    indices = [i for i in channel_set.system_indices(system) if observed[i]]
    if signed:
        return any(diff[i] > thresholds[i] for i in indices)
    return any(abs(diff[i]) > thresholds[i] for i in indices)


def _system_is_quiet(
    diff: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    channel_set: ChannelSet,
    system: ChannelSystem,
    observed: NDArray[np.bool_],
) -> bool:
    """Whether every channel of ``system`` was reported and sits inside its quiet band.

    A system with no reported channel is *unknown*, not quiet: a rule such as
    respiratory distress without fever must not fire when the wearer's thermal
    channel was missing this epoch.
    """
    indices = channel_set.system_indices(system)
    if not any(observed[i] for i in indices):
        return False
    return all(
        abs(diff[i]) < thresholds[i] * channel_set.channels[i].quiet_fraction
        for i in indices
        if observed[i]
    )


def classify_anomaly(
    observation: NDArray[np.float64],
    baseline: NDArray[np.float64],
    channel_set: ChannelSet = DEFAULT_CHANNEL_SET,
    observed: NDArray[np.bool_] | None = None,
) -> AnomalyType | None:
    """Classify the type of anomaly based on deviation pattern.

    Rules are written against physiological *systems* and each channel's own
    ``deviation_threshold``, so adding channels does not require new rules:

    - Respiratory: a respiratory channel is elevated while the thermal
      channels stay inside their quiet band (distress without fever)
    - Multi-system: at least ``MULTI_SYSTEM_MIN_CHANNELS`` channels exceeded
    - Febrile: a thermal channel is elevated
    - Cardiac: a cardiac channel deviates in either direction

    ``observed`` marks the channels the device reported this epoch. Channels the
    device did not report are neither exceeded nor quiet, so a duty-cycled
    channel can neither invent an excursion nor satisfy a "stayed quiet" arm.
    """
    diff = observation - baseline
    thresholds = channel_set.deviation_thresholds
    mask = np.ones(len(channel_set), dtype=np.bool_) if observed is None else observed
    exceeded = int(np.count_nonzero(mask & (np.abs(diff) > thresholds)))

    if exceeded == 0:
        return None
    respiratory_up = _system_exceeded(
        diff, thresholds, channel_set, ChannelSystem.RESPIRATORY, mask, signed=True
    )
    if respiratory_up and _system_is_quiet(
        diff, thresholds, channel_set, ChannelSystem.THERMAL, mask
    ):
        return AnomalyType.RESPIRATORY
    if exceeded >= MULTI_SYSTEM_MIN_CHANNELS:
        return AnomalyType.MULTI_SYSTEM
    if _system_exceeded(diff, thresholds, channel_set, ChannelSystem.THERMAL, mask, signed=True):
        return AnomalyType.FEBRILE
    if _system_exceeded(diff, thresholds, channel_set, ChannelSystem.CARDIAC, mask):
        return AnomalyType.CARDIAC
    return AnomalyType.MULTI_SYSTEM
