"""Device-side exposure advisories assembled from protocol-visible signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np

from garland.constants import STEPS_PER_DAY
from garland.privacy import AnomalyType, BroadcastQuery, noised_aggregate_count

AdvisoryKey = tuple[AnomalyType, frozenset[int]]


@dataclass
class AdvisoryConfig:
    """Configuration for local advisories and opt-in confirmation."""

    enabled: bool = False
    clinic_visit_rate_per_day: float = 0.3
    advisory_expiry_steps: int = STEPS_PER_DAY
    advisory_confirmation_epsilon: float = 0.05
    tier2_confirmations: int = 3
    tier3_confirmations: int = 10

    def __post_init__(self) -> None:
        """Validate advisory rates, privacy cost, and ordered tiers."""
        if not isfinite(self.clinic_visit_rate_per_day) or self.clinic_visit_rate_per_day < 0.0:
            raise ValueError("clinic_visit_rate_per_day must be finite and non-negative")
        if self.advisory_expiry_steps <= 0:
            raise ValueError("advisory_expiry_steps must be positive")
        if (
            not isfinite(self.advisory_confirmation_epsilon)
            or self.advisory_confirmation_epsilon <= 0.0
        ):
            raise ValueError("advisory_confirmation_epsilon must be finite and positive")
        if self.tier2_confirmations <= 0:
            raise ValueError("tier2_confirmations must be positive")
        if self.tier3_confirmations <= self.tier2_confirmations:
            raise ValueError("tier3_confirmations must exceed tier2_confirmations")


@dataclass
class Advisory:
    """One device-local advisory refreshed by a matching broadcast."""

    key: AdvisoryKey
    hypothesis: AnomalyType
    estimated_exposure_step: int | None
    tier: int
    issued_step: int
    last_refresh_step: int
    clinic_visited: bool = False


@dataclass
class AdvisoryStepResult:
    """Measurement-visible results from one advisory protocol step."""

    issued: list[tuple[int, Advisory]] = field(default_factory=list)
    clinic_visits: int = 0
    confirmations_by_type: dict[str, int] = field(default_factory=dict)
    released_counts: dict[AdvisoryKey, int] = field(default_factory=dict)


class AdvisoryEngine:
    """Coordinate local advisory refresh, opt-in clinics, and public counts."""

    def __init__(self, config: AdvisoryConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.confirmed_by_key: dict[AdvisoryKey, int] = {}
        self.published_by_key: dict[AdvisoryKey, int] = {}
        self._released_confirmed_by_key: dict[AdvisoryKey, int] = {}
        self.confirmed_by_type: dict[str, int] = {"disease": 0, "toxin": 0}
        self._zone_population_by_key: dict[AdvisoryKey, int] = {}
        self._toxin_exposure_history: dict[int, set[int]] = {}

    def refresh(
        self,
        queries: list[BroadcastQuery],
        wearable_agents_by_cell: dict[int, list[Any]],
        step: int,
    ) -> list[tuple[int, Advisory]]:
        """Refresh matching local advisories from newly issued broadcasts."""
        issued: list[tuple[int, Advisory]] = []
        for query in queries:
            key = (query.anomaly_type, frozenset(query.zone_cells))
            population = sum(
                len(wearable_agents_by_cell.get(cell_id, ())) for cell_id in query.zone_cells
            )
            self._zone_population_by_key[key] = max(1, population)
            for cell_id in query.zone_cells:
                for agent in wearable_agents_by_cell.get(cell_id, ()):
                    advisory = self._refresh_agent(agent, query, key, step)
                    if advisory is not None:
                        issued.append((agent.idx, advisory))
        self._expire(wearable_agents_by_cell, step)
        return issued

    @staticmethod
    def _refresh_agent(
        agent: Any, query: BroadcastQuery, key: AdvisoryKey, step: int
    ) -> Advisory | None:
        """Refresh one matching local device advisory."""
        if (
            not agent.is_operational
            or not agent.anomaly_active
            or agent.anomaly_type != query.anomaly_type
        ):
            return None
        current = agent.advisory
        if current is not None and current.key == key:
            current.last_refresh_step = step
            return None
        advisory = Advisory(
            key=key,
            hypothesis=query.anomaly_type,
            estimated_exposure_step=agent.anomaly_onset_step,
            tier=1,
            issued_step=step,
            last_refresh_step=step,
        )
        agent.advisory = advisory
        return advisory

    def _expire(self, wearable_agents_by_cell: dict[int, list[Any]], step: int) -> None:
        """Expire advisories that received no matching broadcast."""
        for agents in wearable_agents_by_cell.values():
            for agent in agents:
                advisory = agent.advisory
                if (
                    advisory is not None
                    and step - advisory.last_refresh_step >= self.config.advisory_expiry_steps
                ):
                    agent.advisory = None

    def process_step(
        self,
        agents: list[Any],
        step: int,
        disease_exposed: set[int],
        toxin_exposed: set[int],
    ) -> AdvisoryStepResult:
        """Process opt-in clinic visits and release noisy confirmation counts."""
        result = AdvisoryStepResult()
        for agent_idx in toxin_exposed:
            self._toxin_exposure_history.setdefault(agent_idx, set()).add(step)
        active_keys: set[AdvisoryKey] = set()
        for agent in agents:
            advisory = agent.advisory
            if advisory is None:
                continue
            active_keys.add(advisory.key)
            if advisory.clinic_visited or self.rng.random() >= (
                self.config.clinic_visit_rate_per_day / STEPS_PER_DAY
            ):
                continue
            advisory.clinic_visited = True
            result.clinic_visits += 1
            confirmed_type = self._confirmation_type(
                agent.idx,
                disease_exposed,
                self._toxin_exposure_history,
                advisory,
                step,
                self.config.advisory_expiry_steps,
            )
            if confirmed_type is None:
                continue
            self.confirmed_by_key[advisory.key] = self.confirmed_by_key.get(advisory.key, 0) + 1
            self.confirmed_by_type[confirmed_type] += 1
            result.confirmations_by_type[confirmed_type] = (
                result.confirmations_by_type.get(confirmed_type, 0) + 1
            )
        for key in active_keys:
            confirmed = self.confirmed_by_key.get(key, 0)
            if confirmed == self._released_confirmed_by_key.get(key, 0):
                continue
            population = self._zone_population_by_key.get(key, 1)
            published = noised_aggregate_count(
                confirmed,
                population,
                self.config.advisory_confirmation_epsilon,
                self.rng,
            )
            self.published_by_key[key] = published
            self._released_confirmed_by_key[key] = confirmed
            result.released_counts[key] = published
        self._apply_tiers(agents)
        return result

    @staticmethod
    def _confirmation_type(
        agent_idx: int,
        disease_exposed: set[int],
        toxin_exposure_history: dict[int, set[int]],
        advisory: Advisory,
        step: int,
        expiry_steps: int,
    ) -> str | None:
        """Resolve a clinic report using ground truth only at the clinic."""
        if agent_idx in disease_exposed:
            return "disease"
        exposure_steps = toxin_exposure_history.get(agent_idx, set())
        window_end = advisory.issued_step + expiry_steps
        if any(
            advisory.issued_step <= exposed_step <= min(step, window_end)
            for exposed_step in exposure_steps
        ):
            return "toxin"
        return None

    def _apply_tiers(self, agents: list[Any]) -> None:
        """Apply only published noisy counts to matching local advisories."""
        for agent in agents:
            advisory = agent.advisory
            if advisory is None:
                continue
            count = self.published_by_key.get(advisory.key, 0)
            if count >= self.config.tier3_confirmations:
                advisory.tier = 3
            elif count >= self.config.tier2_confirmations:
                advisory.tier = 2
