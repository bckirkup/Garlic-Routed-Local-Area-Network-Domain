"""Second-round hypothesis queries for contextual disambiguation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DisambiguationHypothesis(str, Enum):
    """Open-ended hypotheses that can explain a triggered anomaly cluster."""

    RECENT_ADOPTION = "recent_adoption"
    AMBIENT_HEAT = "ambient_heat"


class DisambiguationScore(str, Enum):
    """Model-side score assigned to an issued disambiguation query."""

    WELL_FOUNDED = "well_founded"
    UNFOUNDED = "unfounded"
    UNSCORED = "unscored"


@dataclass
class DisambiguationTriggerConfig:
    """Protocol-visible shape thresholds for one hypothesis."""

    max_zone_cells: int = 3
    min_persistent_windows: int = 2
    max_confirmed_fraction: float = 0.5
    min_breadth: int = 4
    min_breadth_windows: int = 2
    breadth_ratio: float = 2.0


@dataclass
class DisambiguationConfig:
    """Configuration for optional human-approved hypothesis queries.

    ``answer_rate`` is the probability that a reachable participant approves
    an answer. Approved answers are sampled as yes/no using ``yes_rate`` and
    then pass through the protocol's randomized-response mechanism. A zero
    answer rate produces only acknowledgements and eventual unanswered expiry.
    ``ask_epsilon_budget`` limits asks based on epsilon already spent by the
    disambiguation channel. Because the check occurs immediately before each
    ask, one in-flight ask may overshoot the budget by that ask's cost.
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
    breadth_baseline_alpha: float = 0.05
    ask_epsilon_budget: float = 0.0

    def __post_init__(self) -> None:
        self.enabled_hypotheses = frozenset(
            DisambiguationHypothesis(value) for value in self.enabled_hypotheses
        )
        if self.enabled and not self.enabled_hypotheses:
            raise ValueError("disambiguation.enabled requires at least one enabled hypothesis")
        if self.breadth_baseline_alpha <= 0.0 or self.breadth_baseline_alpha > 1.0:
            raise ValueError("breadth_baseline_alpha must be in (0, 1]")
        if self.ask_epsilon_budget < 0.0:
            raise ValueError("ask_epsilon_budget must be non-negative")
        for hypothesis, threshold in (
            (DisambiguationHypothesis.RECENT_ADOPTION, self.recent_adoption),
            (DisambiguationHypothesis.AMBIENT_HEAT, self.ambient_heat),
        ):
            if threshold.min_breadth_windows < 1:
                raise ValueError(f"{hypothesis.value}.min_breadth_windows must be at least 1")
            if threshold.min_breadth_windows > self.trigger_history_steps:
                raise ValueError(
                    f"{hypothesis.value}.min_breadth_windows "
                    f"({threshold.min_breadth_windows}) cannot exceed "
                    f"trigger_history_steps ({self.trigger_history_steps})"
                )
            if threshold.breadth_ratio <= 0.0:
                raise ValueError(f"{hypothesis.value}.breadth_ratio must be positive")
