"""Configuration for device-local biometric baseline maturation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineMaturationConfig:
    """Prior local history used to mature fleet-start device baselines.

    A zero-day maximum disables the phase.  Equal minimum and maximum values
    provide uniform history; different values draw one integer history length
    per fleet-start device.
    """

    minimum_history_days: int = 0
    maximum_history_days: int = 0
    cadence_steps: int = 1

    def __post_init__(self) -> None:
        """Validate history bounds and the sampling cadence."""
        for name, value in (
            ("minimum_history_days", self.minimum_history_days),
            ("maximum_history_days", self.maximum_history_days),
            ("cadence_steps", self.cadence_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"baseline maturation {name} must be an integer")
        if self.minimum_history_days < 0 or self.maximum_history_days < 0:
            raise ValueError("baseline maturation history days must be non-negative")
        if self.minimum_history_days > self.maximum_history_days:
            raise ValueError(
                "baseline maturation minimum_history_days cannot exceed maximum_history_days"
            )
        if self.cadence_steps < 1:
            raise ValueError("baseline maturation cadence_steps must be at least 1")

    @property
    def enabled(self) -> bool:
        """Whether the pre-scenario learning phase has any history to run."""
        return self.maximum_history_days > 0
