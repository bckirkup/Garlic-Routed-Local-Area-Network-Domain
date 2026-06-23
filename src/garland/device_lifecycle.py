"""Device lifecycle model for wearable edge devices.

Models battery depletion, user power-off, device removal (not worn), and
optional home charging. Separates static device ownership (``has_wearable``)
from per-step operability (``DeviceStatus``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class DeviceStatus(IntEnum):
    """Operational status of a wearable device."""

    ACTIVE = 0
    POWERED_OFF = 1
    NOT_WORN = 2
    DEPLETED = 3


@dataclass
class DeviceLifecycleConfig:
    """Configuration for wearable device lifecycle simulation.

    When ``enabled`` is False, all wearables remain ACTIVE for the full run
    (backward-compatible default).
    """

    enabled: bool = False

    # Battery
    battery_enabled: bool = True
    battery_capacity: float = 1.0
    drain_per_step: float = 0.0005
    activity_drain_multiplier: float = 2.0
    home_charge_rate: float = 0.01
    home_charge_enabled: bool = True
    home_radius: float = 100.0

    # Removal (not worn)
    removal_enabled: bool = True
    removal_prob_sleep: float = 0.15
    removal_prob_wake: float = 0.002
    redon_prob: float = 0.08

    # Power-off (user choice)
    power_off_enabled: bool = True
    power_off_prob_night: float = 0.05
    power_on_prob_morning: float = 0.10


def _is_sleep_hour(hour_of_day: float) -> bool:
    """Return True during typical sleep hours (22:00–06:00)."""
    return hour_of_day >= 22.0 or hour_of_day < 6.0


def _is_morning_hour(hour_of_day: float) -> bool:
    """Return True during morning wake window (06:00–10:00)."""
    return 6.0 <= hour_of_day < 10.0


class DeviceLifecycleEngine:
    """Per-step state machine for wearable operability."""

    def __init__(
        self,
        n_wearable: int,
        config: DeviceLifecycleConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.rng = rng
        self.n_wearable = n_wearable
        self.battery_levels = np.full(n_wearable, config.battery_capacity, dtype=np.float64)
        self.status = np.full(n_wearable, DeviceStatus.ACTIVE, dtype=np.int8)

    def is_active(self, local_idx: int) -> bool:
        """Return True if the device at ``local_idx`` is operational."""
        return int(self.status[local_idx]) == DeviceStatus.ACTIVE

    def step(
        self,
        hour_of_day: float,
        activity_level: float,
        at_home_mask: NDArray[np.bool_],
        rng: np.random.Generator | None = None,
    ) -> None:
        """Advance device lifecycle for one 5-minute simulation step."""
        if self.n_wearable == 0:
            return

        gen = rng if rng is not None else self.rng
        cfg = self.config
        n = self.n_wearable
        active = self.status == DeviceStatus.ACTIVE

        # 1. Battery drain for active devices
        if cfg.battery_enabled:
            drain = np.full(n, cfg.drain_per_step, dtype=np.float64)
            if activity_level > 0:
                drain += cfg.drain_per_step * cfg.activity_drain_multiplier * activity_level
            self.battery_levels[active] -= drain[active]
            self.battery_levels = np.clip(self.battery_levels, 0.0, cfg.battery_capacity)

            depleted_now = active & (self.battery_levels <= 0.0)
            self.status[depleted_now] = DeviceStatus.DEPLETED

        # Refresh masks after depletion transitions
        active = self.status == DeviceStatus.ACTIVE
        not_worn = self.status == DeviceStatus.NOT_WORN
        powered_off = self.status == DeviceStatus.POWERED_OFF
        is_sleep = _is_sleep_hour(hour_of_day)
        is_morning = _is_morning_hour(hour_of_day)

        # 2. Removal transitions
        if cfg.removal_enabled:
            if is_sleep:
                remove_roll = gen.random(n) < cfg.removal_prob_sleep
                self.status[active & remove_roll] = DeviceStatus.NOT_WORN
            else:
                redon_roll = gen.random(n) < cfg.redon_prob
                self.status[not_worn & redon_roll] = DeviceStatus.ACTIVE
                if not is_sleep:
                    remove_roll = gen.random(n) < cfg.removal_prob_wake
                    active = self.status == DeviceStatus.ACTIVE
                    self.status[active & remove_roll] = DeviceStatus.NOT_WORN

        # 3. Power-off toggles
        if cfg.power_off_enabled:
            active = self.status == DeviceStatus.ACTIVE
            powered_off = self.status == DeviceStatus.POWERED_OFF
            if is_sleep:
                off_roll = gen.random(n) < cfg.power_off_prob_night
                self.status[active & off_roll] = DeviceStatus.POWERED_OFF
            if is_morning:
                on_roll = gen.random(n) < cfg.power_on_prob_morning
                self.status[powered_off & on_roll] = DeviceStatus.ACTIVE

        # 4. Home charging and recovery from depletion
        if cfg.battery_enabled and cfg.home_charge_enabled and is_sleep:
            chargeable = (self.status == DeviceStatus.DEPLETED) & at_home_mask
            if np.any(chargeable):
                self.battery_levels[chargeable] = np.minimum(
                    cfg.battery_capacity,
                    self.battery_levels[chargeable] + cfg.home_charge_rate,
                )
                recovered = chargeable & (self.battery_levels >= cfg.battery_capacity)
                self.status[recovered] = DeviceStatus.ACTIVE

    def count_by_status(self) -> dict[str, int]:
        """Return counts per device status for metrics."""
        return {
            "active": int(np.sum(self.status == DeviceStatus.ACTIVE)),
            "powered_off": int(np.sum(self.status == DeviceStatus.POWERED_OFF)),
            "not_worn": int(np.sum(self.status == DeviceStatus.NOT_WORN)),
            "depleted": int(np.sum(self.status == DeviceStatus.DEPLETED)),
        }
