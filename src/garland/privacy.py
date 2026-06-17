"""Decentralized Privacy & Routing Protocol for GARLAND.

Implements the broadcast-and-filter differential privacy framework:
- Blind Gating (homomorphic encryption simulation)
- Secure Threshold Aggregator
- Spatial Dilution (K-Anonymity)
- Reverse-Query Broadcast
- Uplink Perturbation (Local DP via Randomized Response + Planar Laplace)
- Traffic Obfuscation (dummy noise packets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class AnomalyType(Enum):
    """Types of biometric anomalies detected by agents."""

    RESPIRATORY = "respiratory"
    CARDIAC = "cardiac"
    FEBRILE = "febrile"
    MULTI_SYSTEM = "multi_system"


class EncryptedToken(NamedTuple):
    """Simulated homomorphically encrypted anomaly report.

    In production this would be a Paillier/BFV ciphertext.
    Here we simulate the protocol semantics.
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
        Privacy budget consumed per randomized response.
    randomized_response_p : float
        Probability of truthful response in coin-flip RR.
    laplace_scale : float
        Scale parameter for Planar Laplace mechanism (meters).
    dummy_rate : float
        Rate at which non-matching agents emit dummy packets.
    """

    threshold_m: int = 5
    k_min: int = 50
    time_window_steps: int = 12
    epsilon_per_response: float = 0.1
    randomized_response_p: float = 0.75
    laplace_scale: float = 200.0
    dummy_rate: float = 0.01


@dataclass
class AggregatorState:
    """State of the Secure Threshold Aggregator.

    Cannot read individual tokens — only counts matching patterns.
    """

    # Zone → AnomalyType → list of timestamps
    token_counts: dict[int, dict[AnomalyType, list[int]]] = field(default_factory=dict)
    # Active broadcasts
    active_queries: list[BroadcastQuery] = field(default_factory=list)
    # Responses collected
    responses: list[PerturbedResponse] = field(default_factory=list)
    # Cumulative privacy budget
    total_epsilon: float = 0.0
    # History of epsilon expenditure per step
    epsilon_history: list[float] = field(default_factory=list)

    def receive_token(self, token: EncryptedToken) -> None:
        """Ingest an encrypted token (additive homomorphic sum)."""
        if token.is_dummy:
            return  # Dummy packets are filtered by the aggregator
        zone = token.zone_id
        atype = token.anomaly_type
        if zone not in self.token_counts:
            self.token_counts[zone] = {}
        if atype not in self.token_counts[zone]:
            self.token_counts[zone][atype] = []
        self.token_counts[zone][atype].append(token.timestamp_bin)

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

    def record_epsilon(self, epsilon: float) -> None:
        """Track privacy budget expenditure."""
        self.total_epsilon += epsilon
        self.epsilon_history.append(self.total_epsilon)


@dataclass
class BroadcastQuery:
    """Reverse-query broadcast sent to devices in a dilated zone."""

    zone_cells: list[int]  # Dilated zone (after K-anonymity expansion)
    anomaly_type: AnomalyType
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


def planar_laplace_noise(scale: float, rng: np.random.Generator) -> tuple[float, float]:
    """Generate 2D Planar Laplace noise for geo-indistinguishability.

    The Planar Laplace mechanism guarantees ε-geo-indistinguishability:
    any two points within distance d have probability ratio ≤ e^(εd).

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
    # This produces the 2D Laplace (optimal for geo-indistinguishability)
    theta = rng.uniform(0, 2 * np.pi)
    # Inverse CDF method for Gamma(2, scale)
    r = -scale * (np.log(rng.random()) + np.log(rng.random()))
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return (float(dx), float(dy))


def randomized_response(true_value: bool, p: float, rng: np.random.Generator) -> bool:
    """Coin-flip randomized response mechanism.

    With probability p, report truthfully. Otherwise, flip a fair coin.
    Provides plausible deniability: ε = ln((p + 0.5*(1-p)) / (0.5*(1-p))).

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


def compute_adaptive_composition_epsilon(
    n_queries: int, epsilon_per_query: float, delta: float = 1e-6
) -> float:
    """Compute total privacy loss under advanced composition theorem.

    Uses the optimal composition bound:
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


def classify_anomaly(
    observation: NDArray[np.float64],
    baseline: NDArray[np.float64],
) -> AnomalyType | None:
    """Classify the type of anomaly based on deviation pattern.

    Uses the pattern of deviations to distinguish:
    - Respiratory: primarily RR elevation without fever
    - Febrile: temperature + HR elevation (suggests infection)
    - Cardiac: isolated HR/HRV anomaly
    - Multi-system: multiple parameters elevated
    """
    diff = observation - baseline
    hr_dev = diff[0]
    hrv_dev = diff[1]
    rr_dev = diff[2]
    temp_dev = diff[3]

    anomalies = 0
    if abs(hr_dev) > 10:
        anomalies += 1
    if abs(hrv_dev) > 10:
        anomalies += 1
    if abs(rr_dev) > 4:
        anomalies += 1
    if abs(temp_dev) > 0.8:
        anomalies += 1

    if anomalies == 0:
        return None
    if anomalies >= 3:
        return AnomalyType.MULTI_SYSTEM
    if rr_dev > 4 and abs(temp_dev) < 0.5:
        return AnomalyType.RESPIRATORY
    if temp_dev > 0.8:
        return AnomalyType.FEBRILE
    if abs(hr_dev) > 10 or abs(hrv_dev) > 10:
        return AnomalyType.CARDIAC
    return AnomalyType.MULTI_SYSTEM
