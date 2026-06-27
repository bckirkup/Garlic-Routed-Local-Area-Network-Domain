"""Hazard Engine: SEIR infectious disease and environmental toxin plume.

Implements:
- Point-source Gaussian plume dispersion (chlorine/chemical leak model)
- SEIR compartmental model with spatial proximity-based transmission
- Biometric perturbation functions for each hazard type
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class SEIRState(IntEnum):
    """SEIR compartmental states."""

    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTIOUS = 2
    RECOVERED = 3


@dataclass
class OutbreakSeed:
    """Timed, optionally localized seeding event for a disease outbreak wave.

    Parameters
    ----------
    outbreak_id : str
        Identifier for metrics and detection attribution.
    start_step : int
        Simulation step when seed infections are introduced.
    initial_infected : int
        Number of agents seeded as infectious at ``start_step``.
    center_x : float | None
        Optional outbreak center X (meters). ``None`` seeds globally at random.
    center_y : float | None
        Optional outbreak center Y (meters).
    seed_radius : float
        Radius (meters) around ``center_x/center_y`` for localized seeding.
    """

    outbreak_id: str = "outbreak_0"
    start_step: int = 0
    initial_infected: int = 10
    center_x: float | None = None
    center_y: float | None = None
    seed_radius: float = 500.0


@dataclass
class SEIRConfig:
    """SEIR model parameters grounded in COVID-19/Influenza benchmarks.

    Parameters
    ----------
    beta : float
        Transmission rate per infectious contact per 5-min step.
        Calibrated for R0 ≈ 2.5 with ~8 contacts/step within 2m.
    sigma : float
        Rate of E→I transition (1/incubation_period_steps).
        Default: 1/(5 days * 288 steps/day) = ~0.000694/step.
    gamma : float
        Rate of I→R transition (1/infectious_period_steps).
        Default: 1/(10 days * 288 steps/day) = ~0.000347/step.
    contact_radius : float
        Spatial radius (meters) for transmission contacts.
    initial_infected : int
        Seed cases at simulation start (legacy mode when ``outbreaks`` is empty).
    outbreaks : list[OutbreakSeed]
        Timed outbreak waves. When non-empty, overrides ``initial_infected`` seeding.
    max_infectious_checks : int
        Cap on infectious agents sampled for proximity transmission per step.
        Keeps S→E contact search bounded at city scale; increase for higher
        fidelity when the infectious fraction is large.
    """

    beta: float = 0.015
    sigma: float = 0.000694
    gamma: float = 0.000347
    contact_radius: float = 2.0
    initial_infected: int = 10
    outbreaks: list[OutbreakSeed] = field(default_factory=list)
    max_infectious_checks: int = 500


@dataclass
class PlumeConfig:
    """Point-source Gaussian plume model parameters.

    Based on industrial safety thresholds for chlorine gas dispersion.

    Parameters
    ----------
    source_x : float
        Plume source X coordinate (meters).
    source_y : float
        Plume source Y coordinate (meters).
    release_rate : float
        Mass release rate (kg/s).
    wind_speed : float
        Ambient wind speed (m/s).
    wind_direction : float
        Wind direction in radians (0 = East, π/2 = North).
    stability_class : str
        Pasquill-Gifford stability class (A-F). Default 'D' (neutral).
    start_step : int
        Simulation step when plume begins.
    duration_steps : int
        Number of steps the plume persists.
    plume_id : str
        Identifier for metrics and detection attribution.
    """

    plume_id: str = "plume_0"
    source_x: float = 5000.0
    source_y: float = 5000.0
    release_rate: float = 1.0
    wind_speed: float = 3.0
    wind_direction: float = 0.0
    stability_class: str = "D"
    start_step: int = 288  # Start at step 288 (24h into sim)
    duration_steps: int = 144  # Lasts 12 hours


# Pasquill-Gifford dispersion coefficients (simplified)
_PG_COEFFICIENTS: dict[str, tuple[float, float, float, float]] = {
    # (σy_a, σy_b, σz_a, σz_b) where σ = a * x^b
    "A": (0.22, 0.894, 0.20, 0.894),
    "B": (0.16, 0.894, 0.12, 0.894),
    "C": (0.11, 0.894, 0.08, 0.894),
    "D": (0.08, 0.894, 0.06, 0.894),
    "E": (0.06, 0.894, 0.03, 0.894),
    "F": (0.04, 0.894, 0.016, 0.894),
}


def compute_plume_concentration(
    agent_x: NDArray[np.float32],
    agent_y: NDArray[np.float32],
    config: PlumeConfig,
    current_step: int,
) -> NDArray[np.float64]:
    """Compute toxin concentration at each agent position.

    Uses a Gaussian plume model with Pasquill-Gifford dispersion.

    Returns concentration in arbitrary units [0, ∞). Values > 1.0
    represent levels above immediate health-effect threshold.
    """
    n = len(agent_x)
    concentrations = np.zeros(n, dtype=np.float64)

    if current_step < config.start_step:
        return concentrations
    if current_step >= config.start_step + config.duration_steps:
        return concentrations

    # Transform to wind-aligned coordinate system
    cos_w = np.cos(config.wind_direction)
    sin_w = np.sin(config.wind_direction)
    dx = agent_x - config.source_x
    dy = agent_y - config.source_y

    # Downwind (x') and crosswind (y') distances
    x_prime = dx * cos_w + dy * sin_w
    y_prime = -dx * sin_w + dy * cos_w

    # Only compute for agents downwind
    downwind_mask = x_prime > 1.0  # At least 1m downwind
    if not np.any(downwind_mask):
        return concentrations

    coeffs = _PG_COEFFICIENTS.get(config.stability_class, _PG_COEFFICIENTS["D"])
    sy_a, sy_b, sz_a, sz_b = coeffs

    x_dw = x_prime[downwind_mask]
    y_cw = y_prime[downwind_mask]

    # Dispersion parameters
    sigma_y = sy_a * np.power(x_dw, sy_b)
    sigma_z = sz_a * np.power(x_dw, sz_b)

    # Gaussian plume formula (ground-level, no effective stack height)
    Q = config.release_rate
    u = config.wind_speed

    c = (Q / (np.pi * u * sigma_y * sigma_z)) * np.exp(
        -0.5 * (y_cw / sigma_y) ** 2
    )

    concentrations[downwind_mask] = c
    return concentrations


def compute_plume_concentrations(
    agent_x: NDArray[np.float32],
    agent_y: NDArray[np.float32],
    plumes: list[PlumeConfig],
    current_step: int,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Compute combined and per-plume toxin concentrations.

    Returns
    -------
    total : NDArray
        Sum of concentrations across all active plume sources.
    per_plume : dict[str, NDArray]
        Concentration field keyed by ``plume_id``.
    """
    n = len(agent_x)
    total = np.zeros(n, dtype=np.float64)
    per_plume: dict[str, NDArray[np.float64]] = {}
    for plume in plumes:
        field = compute_plume_concentration(agent_x, agent_y, plume, current_step)
        per_plume[plume.plume_id] = field
        total += field
    return total, per_plume


@dataclass
class SEIREngine:
    """Manages SEIR state transitions for the agent population.

    Uses vectorized operations for performance at 250K scale.
    """

    config: SEIRConfig = field(default_factory=SEIRConfig)
    states: NDArray[np.int8] = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    exposure_step: NDArray[np.int32] = field(
        default_factory=lambda: np.full(0, -1, dtype=np.int32)
    )
    infection_step: NDArray[np.int32] = field(
        default_factory=lambda: np.full(0, -1, dtype=np.int32)
    )
    outbreak_origin: NDArray[np.object_] = field(
        default_factory=lambda: np.empty(0, dtype=np.object_)
    )
    _seeded_outbreaks: set[str] = field(default_factory=set, repr=False)

    def initialize(
        self,
        n_agents: int,
        rng: np.random.Generator,
        agent_x: NDArray[np.float32] | None = None,
        agent_y: NDArray[np.float32] | None = None,
    ) -> None:
        """Set initial SEIR states and apply step-0 outbreak seeds if configured."""
        self.states = np.full(n_agents, SEIRState.SUSCEPTIBLE, dtype=np.int8)
        self.exposure_step = np.full(n_agents, -1, dtype=np.int32)
        self.infection_step = np.full(n_agents, -1, dtype=np.int32)
        self.outbreak_origin = np.full(n_agents, "", dtype=np.object_)
        self._seeded_outbreaks = set()

        if self.config.outbreaks:
            for outbreak in self.config.outbreaks:
                if outbreak.start_step == 0:
                    self._apply_outbreak_seed(outbreak, 0, agent_x, agent_y, rng)
        else:
            infected_idx = rng.choice(n_agents, self.config.initial_infected, replace=False)
            self.states[infected_idx] = SEIRState.INFECTIOUS
            self.infection_step[infected_idx] = 0
            self.outbreak_origin[infected_idx] = "outbreak_0"

    def maybe_seed_outbreaks(
        self,
        current_step: int,
        agent_x: NDArray[np.float32],
        agent_y: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> None:
        """Introduce outbreak seeds scheduled for ``current_step``."""
        for outbreak in self.config.outbreaks:
            if (
                outbreak.outbreak_id not in self._seeded_outbreaks
                and outbreak.start_step == current_step
            ):
                self._apply_outbreak_seed(outbreak, current_step, agent_x, agent_y, rng)

    def _apply_outbreak_seed(
        self,
        outbreak: OutbreakSeed,
        current_step: int,
        agent_x: NDArray[np.float32] | None,
        agent_y: NDArray[np.float32] | None,
        rng: np.random.Generator,
    ) -> None:
        """Seed infectious agents for a single outbreak wave."""
        if outbreak.initial_infected <= 0:
            self._seeded_outbreaks.add(outbreak.outbreak_id)
            return

        susceptible = np.nonzero(self.states == SEIRState.SUSCEPTIBLE)[0]
        if len(susceptible) == 0:
            self._seeded_outbreaks.add(outbreak.outbreak_id)
            return

        if outbreak.center_x is not None and outbreak.center_y is not None and agent_x is not None:
            dx = agent_x[susceptible] - outbreak.center_x
            dy = agent_y[susceptible] - outbreak.center_y  # type: ignore[index]
            in_radius = susceptible[(dx * dx + dy * dy) <= outbreak.seed_radius**2]
            candidates = in_radius if len(in_radius) > 0 else susceptible
        else:
            candidates = susceptible

        count = min(outbreak.initial_infected, len(candidates))
        chosen = rng.choice(candidates, count, replace=False)
        self.states[chosen] = SEIRState.INFECTIOUS
        self.infection_step[chosen] = current_step
        self.outbreak_origin[chosen] = outbreak.outbreak_id
        self._seeded_outbreaks.add(outbreak.outbreak_id)

    def initial_infectious_count(self) -> int:
        """Return baseline infectious count after initialization (for onset detection)."""
        if self.config.outbreaks:
            return int(
                sum(
                    o.initial_infected
                    for o in self.config.outbreaks
                    if o.start_step == 0
                )
            )
        return self.config.initial_infected

    def step(
        self,
        current_step: int,
        agent_x: NDArray[np.float32],
        agent_y: NDArray[np.float32],
        rng: np.random.Generator,
        current_venue_idx: NDArray[np.int32] | None = None,
        venue_contact_multipliers: list[float] | None = None,
        use_proximity_contacts: bool = True,
        use_venue_contacts: bool = False,
    ) -> None:
        """Advance SEIR model one 5-minute step.

        Uses spatial proximity for transmission (contact within radius) and,
        when enabled, elevated transmission among agents co-located at the same
        structured venue.
        """
        # E → I transitions (stochastic based on sigma)
        exposed_mask = self.states == SEIRState.EXPOSED
        if np.any(exposed_mask):
            transition_prob = self.config.sigma
            transitions = rng.random(np.sum(exposed_mask)) < transition_prob
            exposed_indices = np.nonzero(exposed_mask)[0]
            new_infectious = exposed_indices[transitions]
            self.states[new_infectious] = SEIRState.INFECTIOUS
            self.infection_step[new_infectious] = current_step

        # I → R transitions
        infectious_mask = self.states == SEIRState.INFECTIOUS
        if np.any(infectious_mask):
            transition_prob = self.config.gamma
            transitions = rng.random(np.sum(infectious_mask)) < transition_prob
            infectious_indices = np.nonzero(infectious_mask)[0]
            self.states[infectious_indices[transitions]] = SEIRState.RECOVERED

        # S → E transmission via structured venues and/or spatial proximity
        susceptible_mask = self.states == SEIRState.SUSCEPTIBLE
        infectious_mask = self.states == SEIRState.INFECTIOUS
        if not np.any(infectious_mask) or not np.any(susceptible_mask):
            return

        susceptible_idx = np.nonzero(susceptible_mask)[0]
        infectious_idx = np.nonzero(infectious_mask)[0]
        new_exposed: dict[int, str] = {}

        if (
            use_venue_contacts
            and current_venue_idx is not None
            and venue_contact_multipliers is not None
            and len(venue_contact_multipliers) > 0
        ):
            new_exposed.update(
                self._venue_transmissions(
                    infectious_idx,
                    susceptible_idx,
                    current_venue_idx,
                    venue_contact_multipliers,
                    rng,
                )
            )

        if use_proximity_contacts:
            new_exposed.update(
                self._proximity_transmissions(
                    infectious_idx, susceptible_idx, agent_x, agent_y, rng
                )
            )

        if new_exposed:
            new_exposed_arr = np.array(list(new_exposed.keys()), dtype=np.intp)
            still_susceptible = self.states[new_exposed_arr] == SEIRState.SUSCEPTIBLE
            actual_new = new_exposed_arr[still_susceptible]
            self.states[actual_new] = SEIRState.EXPOSED
            self.exposure_step[actual_new] = current_step
            for idx in actual_new:
                self.outbreak_origin[idx] = new_exposed[int(idx)]

    def _proximity_transmissions(
        self,
        infectious_idx: NDArray[np.intp],
        susceptible_idx: NDArray[np.intp],
        agent_x: NDArray[np.float32],
        agent_y: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> dict[int, str]:
        """S→E transmission via Euclidean proximity."""
        new_exposed: dict[int, str] = {}
        n_infectious = len(infectious_idx)

        max_check = min(n_infectious, self.config.max_infectious_checks)
        if n_infectious > max_check:
            check_idx = rng.choice(infectious_idx, max_check, replace=False)
        else:
            check_idx = infectious_idx

        for i_idx in check_idx:
            origin_id = str(self.outbreak_origin[i_idx]) or "outbreak_0"
            dx = agent_x[susceptible_idx] - agent_x[i_idx]
            dy = agent_y[susceptible_idx] - agent_y[i_idx]
            dist_sq = dx * dx + dy * dy
            in_range = susceptible_idx[dist_sq <= self.config.contact_radius**2]

            if len(in_range) > 0:
                infected = rng.random(len(in_range)) < self.config.beta
                for j_idx in in_range[infected]:
                    new_exposed[int(j_idx)] = origin_id
        return new_exposed

    def _venue_transmissions(
        self,
        infectious_idx: NDArray[np.intp],
        susceptible_idx: NDArray[np.intp],
        current_venue_idx: NDArray[np.int32],
        venue_contact_multipliers: list[float],
        rng: np.random.Generator,
    ) -> dict[int, str]:
        """S→E transmission among agents sharing a structured venue."""
        new_exposed: dict[int, str] = {}
        if len(infectious_idx) == 0 or len(susceptible_idx) == 0:
            return new_exposed

        infectious_venues = current_venue_idx[infectious_idx]
        active_venues = np.unique(infectious_venues[infectious_venues >= 0])
        if len(active_venues) == 0:
            return new_exposed

        susceptible_at_venue: dict[int, NDArray[np.intp]] = {}
        for v_idx in active_venues:
            at_venue = susceptible_idx[current_venue_idx[susceptible_idx] == v_idx]
            if len(at_venue) > 0:
                susceptible_at_venue[int(v_idx)] = at_venue

        n_infectious = len(infectious_idx)
        max_check = min(n_infectious, self.config.max_infectious_checks)
        if n_infectious > max_check:
            check_idx = rng.choice(infectious_idx, max_check, replace=False)
        else:
            check_idx = infectious_idx

        for i_idx in check_idx:
            venue = int(current_venue_idx[i_idx])
            if venue < 0 or venue >= len(venue_contact_multipliers):
                continue
            targets = susceptible_at_venue.get(venue)
            if targets is None or len(targets) == 0:
                continue
            origin_id = str(self.outbreak_origin[i_idx]) or "outbreak_0"
            beta_eff = self.config.beta * venue_contact_multipliers[venue]
            infected = rng.random(len(targets)) < beta_eff
            for j_idx in targets[infected]:
                new_exposed[int(j_idx)] = origin_id
        return new_exposed

    def biometric_perturbation(
        self, agent_idx: int, steps_since_infection: int
    ) -> NDArray[np.float64]:
        """Compute biometric shift from infection.

        Infection causes gradual onset:
        - Incubation (E): subtle HRV depression
        - Infectious (I): fever + elevated HR + elevated RR

        Returns additive perturbation vector [HR, HRV, RR, Temp].
        """
        state = self.states[agent_idx]
        if state == SEIRState.EXPOSED:
            # Subtle HRV depression during incubation
            progress = min(steps_since_infection / (288 * 3), 1.0)  # Over 3 days
            return np.array([0.0, -5.0 * progress, 0.0, 0.0], dtype=np.float64)
        elif state == SEIRState.INFECTIOUS:
            # Full symptomatic: fever, tachycardia, tachypnea
            progress = min(steps_since_infection / (288 * 2), 1.0)  # Ramps over 2 days
            return np.array(
                [15.0 * progress, -15.0 * progress, 5.0 * progress, 1.5 * progress],
                dtype=np.float64,
            )
        return np.zeros(4, dtype=np.float64)


def plume_biometric_perturbation(concentration: float) -> NDArray[np.float64]:
    """Compute biometric shift from toxin exposure.

    Toxin causes immediate respiratory distress WITHOUT fever:
    - Elevated RR (dose-dependent)
    - Mild tachycardia (stress response)
    - HRV depression
    - NO temperature increase (key differentiator from infection)

    Returns additive perturbation vector [HR, HRV, RR, Temp].
    """
    if concentration <= 0.01:
        return np.zeros(4, dtype=np.float64)

    # Sigmoid dose-response
    effect = min(concentration / (concentration + 0.5), 1.0)

    return np.array(
        [
            10.0 * effect,  # HR increase (stress)
            -12.0 * effect,  # HRV depression
            12.0 * effect,  # RR spike (primary symptom)
            0.0,  # No fever
        ],
        dtype=np.float64,
    )
