"""Device adoption schedules for onboarding simulations."""

from __future__ import annotations

from dataclasses import dataclass

from garland.constants import STEPS_PER_DAY


@dataclass
class AdoptionConfig:
    """Schedule for first-time wearable adoption.

    ``all_at_start`` preserves the historical default. ``rollout`` adopts a
    fraction of remaining devices per eligible step, ``trickle`` samples each
    remaining device with ``rate`` probability, and ``cohort`` adopts
    ``cohort_size`` household or venue groups every ``interval_steps``.
    ``initial_adopted_fraction`` establishes the already-adopted population
    before the schedule starts. ``new_device_warmup_steps`` suppresses token
    emission for a newly adopted device while its baseline learns. A positive
    rollout rate uses ``ceil`` and therefore adopts at least one remaining
    device per eligible step, even when the fractional target is below one.
    """

    mode: str = "all_at_start"
    start_step: int = 0
    initial_adopted_fraction: float = 1.0
    onboarding_window_steps: int = STEPS_PER_DAY
    rate: float = 0.0
    cohort_size: int = 1
    interval_steps: int = 1
    new_device_warmup_steps: int = 12
    group_by: str = "household"
    venue_kind: str = "any"
