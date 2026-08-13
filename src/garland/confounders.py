"""Persistent ordinary-life confounder processes for GARLAND."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from garland.perturbations import PerturbationCause, PerturbationContribution


@dataclass
class BackgroundILIConfig:
    """Independent symptomatic ILI episodes with household clustering."""

    enabled: bool = False
    target_prevalence: float = 0.02
    onset_probability_per_step: float | None = None
    duration_min_steps: int = 3 * 288
    duration_max_steps: int = 7 * 288
    duration_steps: int | None = None
    household_secondary_multiplier: float = 4.0
    seasonal_amplitude: float = 0.5
    severity_log_sigma: float = 0.25


@dataclass
class CookingIrritantConfig:
    """Household dinner-time irritant events with heterogeneous responses."""

    enabled: bool = False
    events_per_household_day: float = 0.4
    dinner_start_hour: float = 18.0
    dinner_end_hour: float = 21.0
    event_duration_steps: int = 12
    event_intensity_log_sigma: float = 0.35
    frequency_log_sigma: float = 0.8
    susceptibility_log_sigma: float = 1.15
    home_radius: float = 80.0


class ConfounderEngine:
    """Generate persistent, model-side confounder contributions."""

    def __init__(
        self,
        n_agents: int,
        household_ids: NDArray[np.int64],
        rng: np.random.Generator,
        background_ili: BackgroundILIConfig,
        cooking_irritants: CookingIrritantConfig,
    ) -> None:
        self.n_agents = n_agents
        self.household_ids = household_ids
        bit_generator = type(rng.bit_generator)()
        bit_generator.state = deepcopy(rng.bit_generator.state)
        self.rng = np.random.Generator(bit_generator)
        self.background_ili = background_ili
        self.cooking_irritants = cooking_irritants
        self.n_households = int(np.max(household_ids)) + 1 if len(household_ids) else 0
        self.ili_susceptibility = self._lognormal_trait(
            background_ili.severity_log_sigma,
        )
        self.irritant_susceptibility = self._lognormal_trait(
            cooking_irritants.susceptibility_log_sigma,
        )
        self.ili_active = np.zeros(n_agents, dtype=bool)
        self.ili_remaining = np.zeros(n_agents, dtype=np.int32)
        self.ili_delta = np.zeros((n_agents, 4), dtype=np.float64)
        self.household_frequency = self._lognormal_trait(
            cooking_irritants.frequency_log_sigma,
            size=self.n_households,
        )
        self.cooking_exposed_agents = np.zeros(n_agents, dtype=bool)
        self.cooking_remaining = np.zeros(self.n_households, dtype=np.int32)
        self.cooking_intensity = np.zeros(self.n_households, dtype=np.float64)
        self.cooking_event_seen = np.zeros(self.n_households, dtype=bool)

    @property
    def susceptibility(self) -> NDArray[np.float64]:
        """Backward-compatible alias for the irritant-specific trait."""
        return self.irritant_susceptibility

    def _lognormal_trait(
        self,
        sigma: float,
        *,
        size: int | None = None,
    ) -> NDArray[np.float64]:
        trait_size = size or self.n_agents
        if sigma <= 0:
            return np.ones(trait_size, dtype=np.float64)
        values = self.rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma, size=trait_size)
        return values.astype(np.float64)

    def _seasonal_multiplier(self, day_of_year: int) -> float:
        amplitude = self.background_ili.seasonal_amplitude
        phase = 2 * np.pi * (day_of_year - 15) / 365.0
        return max(0.0, 1.0 + amplitude * np.cos(phase))

    def _start_ili_episodes(self, day_of_year: int) -> None:
        cfg = self.background_ili
        if not cfg.enabled:
            return
        active_by_household = np.bincount(
            self.household_ids[self.ili_active], minlength=self.n_households
        )
        household_clustered = active_by_household[self.household_ids] > 0
        if cfg.onset_probability_per_step is None:
            mean_duration = (
                cfg.duration_steps
                if cfg.duration_steps is not None
                else (cfg.duration_min_steps + cfg.duration_max_steps) / 2
            )
            onset_probability = cfg.target_prevalence / mean_duration
        else:
            onset_probability = cfg.onset_probability_per_step
        probability: NDArray[np.float64] = np.full(
            self.n_agents,
            onset_probability * self._seasonal_multiplier(day_of_year),
            dtype=np.float64,
        )
        probability *= np.where(household_clustered, cfg.household_secondary_multiplier, 1.0)
        candidates = (~self.ili_active) & (self.rng.random(self.n_agents) < probability)
        if not np.any(candidates):
            return
        self.ili_active[candidates] = True
        durations: NDArray[np.int32]
        if cfg.duration_steps is not None:
            durations = np.full(int(np.sum(candidates)), cfg.duration_steps, dtype=np.int32)
        else:
            durations = self.rng.integers(
                cfg.duration_min_steps,
                cfg.duration_max_steps + 1,
                size=int(np.sum(candidates)),
                dtype=np.int32,
            )
        self.ili_remaining[candidates] = durations
        severity = self.rng.lognormal(
            mean=-(cfg.severity_log_sigma**2) / 2,
            sigma=cfg.severity_log_sigma,
            size=int(np.sum(candidates)),
        )
        base = np.array([15.0, -15.0, 5.0, 1.5], dtype=np.float64)
        self.ili_delta[candidates] = (
            severity[:, np.newaxis]
            * self.ili_susceptibility[candidates, np.newaxis]
            * base
        )

    def _step_ili(self, day_of_year: int) -> None:
        if not self.background_ili.enabled:
            return
        self.ili_remaining[self.ili_active] -= 1
        ended = self.ili_active & (self.ili_remaining <= 0)
        self.ili_active[ended] = False
        self.ili_delta[ended] = 0.0
        self._start_ili_episodes(day_of_year)

    def _step_cooking(self, hour_of_day: float) -> None:
        cfg = self.cooking_irritants
        if not cfg.enabled:
            return
        self.cooking_remaining[self.cooking_remaining > 0] -= 1
        active_window = cfg.dinner_start_hour <= hour_of_day < cfg.dinner_end_hour
        if not active_window:
            return
        inactive = self.cooking_remaining <= 0
        steps_in_window = max(1, int((cfg.dinner_end_hour - cfg.dinner_start_hour) * 12))
        probability = cfg.events_per_household_day / steps_in_window
        frequency = self.household_frequency
        starts = inactive & (self.rng.random(self.n_households) < probability * frequency)
        self.cooking_remaining[starts] = cfg.event_duration_steps
        self.cooking_intensity[starts] = self.rng.lognormal(
            mean=-(cfg.event_intensity_log_sigma**2) / 2,
            sigma=cfg.event_intensity_log_sigma,
            size=int(np.sum(starts)),
        )

    def step(self, hour_of_day: float, day_of_year: int) -> None:
        """Advance persistent ILI and household cooking episodes."""
        self._step_ili(day_of_year)
        self._step_cooking(hour_of_day)

    def contributions_for_agent(
        self,
        agent_idx: int,
        agent_x: float,
        agent_y: float,
        home_x: float,
        home_y: float,
    ) -> tuple[PerturbationContribution, ...]:
        contributions: list[PerturbationContribution] = []
        if self.ili_active[agent_idx]:
            contributions.append(
                PerturbationContribution(
                    PerturbationCause.BACKGROUND_ILI,
                    self.ili_delta[agent_idx],
                )
            )
        household_id = int(self.household_ids[agent_idx])
        if self.cooking_remaining[household_id] > 0:
            cfg = self.cooking_irritants
            at_home = (agent_x - home_x) ** 2 + (agent_y - home_y) ** 2 <= cfg.home_radius**2
            if at_home:
                delta = (
                    self.cooking_intensity[household_id]
                    * self.irritant_susceptibility[agent_idx]
                    * np.array([10.0, -12.0, 12.0, 0.0], dtype=np.float64)
                )
                contributions.append(
                    PerturbationContribution(PerturbationCause.IRRITANT_EXPOSURE, delta)
                )
        return tuple(contributions)

    def record_cooking_exposures(
        self,
        agent_x: NDArray[np.float32],
        agent_y: NDArray[np.float32],
        home_x: NDArray[np.float32],
        home_y: NDArray[np.float32],
    ) -> tuple[int, int, int]:
        """Record exposure and first-step reach for active cooking events."""
        if not self.cooking_irritants.enabled:
            return 0, 0, 0
        active_households = self.cooking_remaining > 0
        if not np.any(active_households):
            self.cooking_event_seen[~active_households] = False
            return 0, 0, 0
        active_agents = active_households[self.household_ids]
        distance_sq = (agent_x - home_x[self.household_ids]) ** 2 + (
            agent_y - home_y[self.household_ids]
        ) ** 2
        at_home = distance_sq <= self.cooking_irritants.home_radius**2
        self.cooking_exposed_agents |= active_agents & at_home
        new_events = active_households & ~self.cooking_event_seen
        event_households = np.flatnonzero(new_events)
        reached = 0
        members = 0
        for household_id in event_households:
            member_mask = self.household_ids == household_id
            members += int(np.sum(member_mask))
            reached += int(np.sum(member_mask & at_home))
        self.cooking_event_seen[active_households] = True
        self.cooking_event_seen[~active_households] = False
        return len(event_households), members, reached
