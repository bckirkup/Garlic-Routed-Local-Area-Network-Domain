"""Sequential per-person anomaly detection state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.channels import DEFAULT_CHANNEL_SET, ChannelSet


@dataclass
class SequentialDetector:
    """CUSUM detector with latched alarms and residual-pattern memory."""

    reference_value: float = 2.0
    threshold: float = 10.0
    clear_steps: int = 3
    clear_fraction: float = 0.5
    residual_ewma_alpha: float = 0.2
    statistic: float = 0.0
    zero_steps: int = 0
    alarm_active: bool = False
    channel_set: ChannelSet = DEFAULT_CHANNEL_SET
    residual_ewma: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.residual_ewma = self.channel_set.zeros()

    def reset(self) -> None:
        """Discard accumulated sequential state."""
        self.statistic = 0.0
        self.zero_steps = 0
        self.alarm_active = False
        self.residual_ewma.fill(0.0)

    def update(
        self,
        distance: float,
        residual: NDArray[np.float64],
    ) -> bool:
        """Update the detector and return whether a new alarm started."""
        self.statistic = max(0.0, self.statistic + distance - self.reference_value)
        self.residual_ewma = (
            1.0 - self.residual_ewma_alpha
        ) * self.residual_ewma + self.residual_ewma_alpha * residual

        if self.alarm_active:
            if self.statistic <= self.threshold * self.clear_fraction:
                self.zero_steps += 1
                if self.zero_steps >= self.clear_steps:
                    self.reset()
            else:
                self.zero_steps = 0
            return False

        if self.statistic > self.threshold:
            self.alarm_active = True
            self.zero_steps = 0
            return True
        return False


__all__ = ["SequentialDetector"]
