"""Second-round hypothesis queries for contextual disambiguation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DisambiguationHypothesis(str, Enum):
    """Open-ended hypotheses that can explain a triggered anomaly cluster."""

    RECENT_ADOPTION = "recent_adoption"
    AMBIENT_HEAT = "ambient_heat"


@dataclass
class DisambiguationTriggerConfig:
    """Protocol-visible shape thresholds for one hypothesis."""

    max_zone_cells: int = 3
    min_persistent_windows: int = 2
    max_confirmed_fraction: float = 0.5
    min_breadth: int = 4


@dataclass
class DisambiguationConfig:
    """Configuration for optional human-approved hypothesis queries.

    ``answer_rate`` is the probability that a reachable participant approves
    an answer. Approved answers are sampled as yes/no using ``yes_rate`` and
    then pass through the protocol's randomized-response mechanism. A zero
    answer rate produces only acknowledgements and eventual unanswered expiry.
    """

    enabled: bool = False
    enabled_hypotheses: frozenset[DisambiguationHypothesis] = frozenset()
    recent_adoption: DisambiguationTriggerConfig = field(
        default_factory=DisambiguationTriggerConfig
    )
    ambient_heat: DisambiguationTriggerConfig = field(default_factory=DisambiguationTriggerConfig)
    trigger_history_steps: int = 4
    answer_rate: float = 0.5
    yes_rate: float = 0.5
    expiry_steps: int = 12
    ack_noise_scale: float = 1.0
    ack_epsilon: float = 0.01

    def __post_init__(self) -> None:
        self.enabled_hypotheses = frozenset(
            DisambiguationHypothesis(value) for value in self.enabled_hypotheses
        )
        if self.enabled and not self.enabled_hypotheses:
            raise ValueError("disambiguation.enabled requires at least one enabled hypothesis")
