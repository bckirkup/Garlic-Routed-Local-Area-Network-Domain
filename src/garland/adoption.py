"""Device adoption schedules for onboarding simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdoptionConfig:
    """Schedule for first-time wearable adoption.

    ``all_at_start`` preserves the historical default. ``rollout`` adopts a
    fraction of remaining devices per eligible step, ``trickle`` samples each
    remaining device with ``rate`` probability, and ``cohort`` adopts
    ``cohort_size`` household or venue groups every ``interval_steps``.
    """

    mode: str = "all_at_start"
    start_step: int = 0
    rate: float = 0.0
    cohort_size: int = 1
    interval_steps: int = 1
    group_by: str = "household"
