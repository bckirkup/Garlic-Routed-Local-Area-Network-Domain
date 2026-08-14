"""Second-round hypothesis queries for contextual disambiguation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DisambiguationHypothesis(str, Enum):
    """Open-ended hypotheses that can explain a triggered anomaly cluster."""

    RECENT_ADOPTION = "recent_adoption"


@dataclass
class DisambiguationConfig:
    """Configuration for optional human-approved hypothesis queries.

    ``answer_rate`` is the probability that a reachable participant approves
    an answer. Approved answers are sampled as yes/no using ``yes_rate`` and
    then pass through the protocol's randomized-response mechanism. A zero
    answer rate produces only acknowledgements and eventual unanswered expiry.
    """

    enabled: bool = False
    hypothesis: DisambiguationHypothesis = DisambiguationHypothesis.RECENT_ADOPTION
    min_onboarding_wearables_in_zone: int = 1
    answer_rate: float = 0.5
    yes_rate: float = 0.5
    expiry_steps: int = 12
    ack_noise_scale: float = 1.0
    ack_epsilon: float = 0.01
