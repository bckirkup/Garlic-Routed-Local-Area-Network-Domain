"""Benign, cause-labelled biometric confounder sources."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.constants import STEPS_PER_DAY
from garland.perturbations import PerturbationCause, PerturbationContribution
from garland.venues import VenueEngine, VenueType


@dataclass
class ConfoundersConfig:
    """Configuration for independent and heat-wave confounder sources."""

    enabled: bool = False
    exercise_rate: float = 0.01
    exercise_duration_steps: int = 6
    exercise_hr_delta: float = 8.0
    exercise_hrv_delta: float = -6.0
    exercise_temperature_delta: float = 0.08
    sleep_disruption_rate: float = 0.05
    sleep_disruption_delay_steps: int = 96
    sleep_disruption_duration_steps: int = 12
    sleep_disruption_hr_delta: float = 6.0
    sleep_disruption_hrv_delta: float = -5.0
    sleep_disruption_temperature_delta: float = 0.05
    sensor_artifact_probability: float = 0.25
    sensor_artifact_hr_delta: float = 14.0
    sensor_artifact_hrv_delta: float = -12.0
    sensor_artifact_temperature_delta: float = 0.25
    heat_wave_start_step: int = 0
    heat_wave_duration_steps: int = 0
    heat_wave_hr_delta: float = 5.0
    heat_wave_hrv_delta: float = 0.0
    heat_wave_temperature_delta: float = 0.8
    heat_wave_amplitude_jitter: float = 0.1
    venue_crowding_rate: float = 0.0
    venue_crowding_duration_steps: int = 12
    venue_crowding_venue_types: tuple[VenueType, ...] = (
        VenueType.THIRD_PLACE,
        VenueType.GATHERING,
    )
    venue_crowding_occupancy_reference: float = 20.0
    venue_crowding_hr_delta: float = 8.0
    venue_crowding_hrv_delta: float = -6.0
    venue_crowding_temperature_delta: float = 0.15
    venue_crowding_amplitude_jitter: float = 0.1
    background_ili_daily_incidence: float = 0.0
    background_ili_secondary_probability: float = 0.0
    background_ili_incubation_delay_steps: int = 12
    background_ili_symptomatic_duration_steps: int = 24
    background_ili_hr_delta: float = 10.0
    background_ili_hrv_delta: float = -8.0
    background_ili_temperature_delta: float = 0.7
    background_ili_amplitude_jitter: float = 0.1


@dataclass(frozen=True)
class HeatWaveInstance:
    """Model-side identity and footprint of one heat-wave episode."""

    instance_id: str
    start_step: int
    end_step: int
    zone_ids: tuple[int, ...]


@dataclass
class BenignInstance:
    """Model-side registry entry for an active correlated benign source."""

    instance_id: str
    cause: PerturbationCause
    start_step: int
    end_step: int
    current_agents: set[int] = field(default_factory=set)
    global_scope: bool = False


@dataclass
class ConfounderStep:
    """Per-step confounder contributions and source activity."""

    contributions: dict[int, tuple[PerturbationContribution, ...]]
    affected_agents_by_cause: dict[PerturbationCause, set[int]]
    heat_wave_active: bool = False
    heat_wave_instance_id: str | None = None
    heat_wave_start_step: int | None = None
    heat_wave_end_step: int | None = None
    benign_instances: dict[str, BenignInstance] = field(default_factory=dict)


class ConfounderEngine:
    """Generate seeded benign perturbations independently of hazard state."""

    def __init__(
        self,
        n_agents: int,
        config: ConfoundersConfig,
        rng: np.random.Generator,
        zone_ids: tuple[int, ...] = (),
        household_ids: NDArray[np.int64] | None = None,
        venue_engine: VenueEngine | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.config = config
        self.rng = rng
        self.zone_ids = zone_ids
        self.household_ids = household_ids
        self.venue_engine = venue_engine
        self.exercise_remaining = np.zeros(n_agents, dtype=np.int32)
        self.sleep_delay = np.zeros(n_agents, dtype=np.int32)
        self.sleep_remaining = np.zeros(n_agents, dtype=np.int32)
        self.sensor_active = np.zeros(n_agents, dtype=bool)
        self.heat_wave_instances = self._build_heat_wave_instances()
        self.heat_wave_amplitudes = np.ones(n_agents, dtype=np.float64)
        self.heat_wave_instance_id: str | None = None
        self.benign_instances: dict[str, BenignInstance] = {}
        self._venue_active: dict[int, BenignInstance] = {}
        self._venue_amplitudes: dict[str, NDArray[np.float64]] = {}
        self._venue_sequence: dict[int, int] = {}
        self._ili_delay = np.zeros(n_agents, dtype=np.int32)
        self._ili_remaining = np.zeros(n_agents, dtype=np.int32)
        self._ili_instance_by_agent: dict[int, str] = {}
        self._ili_amplitudes: dict[str, dict[int, float]] = {}
        self._ili_sequence = 0
        self._instance_expiry: list[tuple[int, str]] = []
        self._active_instance_ids: set[str] = set()

    def _register_instance(self, instance: BenignInstance) -> None:
        self.benign_instances[instance.instance_id] = instance
        heapq.heappush(
            self._instance_expiry, (instance.end_step, instance.instance_id)
        )

    def _prune_instances(self, current_step: int) -> None:
        while self._instance_expiry and self._instance_expiry[0][0] <= current_step:
            _, instance_id = heapq.heappop(self._instance_expiry)
            if instance_id not in self.benign_instances:
                continue
            del self.benign_instances[instance_id]
            self._venue_amplitudes.pop(instance_id, None)
            self._ili_amplitudes.pop(instance_id, None)
            self._active_instance_ids.discard(instance_id)

    def _build_heat_wave_instances(self) -> list[HeatWaveInstance]:
        cfg = self.config
        if cfg.heat_wave_duration_steps <= 0:
            return []
        return [
            HeatWaveInstance(
                instance_id="heat_0",
                start_step=cfg.heat_wave_start_step,
                end_step=cfg.heat_wave_start_step + cfg.heat_wave_duration_steps,
                zone_ids=self.zone_ids,
            )
        ]

    def step(
        self,
        current_step: int,
        hour_of_day: float,
        wearable_mask: NDArray[np.bool_],
        transition_indices: set[int] | None = None,
    ) -> ConfounderStep:
        """Advance sources and return this step's labelled contributions."""
        if not self.config.enabled:
            return ConfounderStep({}, {})

        cfg = self.config
        contributions: dict[int, list[PerturbationContribution]] = {}
        affected: dict[PerturbationCause, set[int]] = {}
        self._prune_instances(current_step)
        for instance_id in self._active_instance_ids:
            instance = self.benign_instances.get(instance_id)
            if (
                instance is not None
                and instance.cause == PerturbationCause.BACKGROUND_ILI
            ):
                instance.current_agents.clear()
        self._active_instance_ids.clear()

        def add(
            agent_idx: int,
            cause: PerturbationCause,
            delta: NDArray[np.float64],
        ) -> None:
            if not wearable_mask[agent_idx]:
                return
            contributions.setdefault(agent_idx, []).append(
                PerturbationContribution(cause, delta)
            )
            affected.setdefault(cause, set()).add(agent_idx)

        active_exercise = self.exercise_remaining > 0
        exercise_weight = 0.25 + 0.75 * max(
            0.0, np.sin(np.pi * (hour_of_day - 6.0) / 12.0)
        ) if 6.0 <= hour_of_day <= 18.0 else 0.1
        exercise_onsets = (
            self.rng.random(self.n_agents) < cfg.exercise_rate * exercise_weight
        )
        exercise_onsets &= ~active_exercise & wearable_mask
        self.exercise_remaining[exercise_onsets] = max(1, cfg.exercise_duration_steps)
        active_exercise = self.exercise_remaining > 0
        exercise_delta = np.array(
            [cfg.exercise_hr_delta, cfg.exercise_hrv_delta, 0.0, cfg.exercise_temperature_delta],
            dtype=np.float64,
        )
        for idx in np.flatnonzero(active_exercise):
            add(int(idx), PerturbationCause.EXERCISE, exercise_delta)
        self.exercise_remaining[active_exercise] -= 1

        nightly_step = current_step % STEPS_PER_DAY == 22 * 12
        if nightly_step:
            disruptions = (
                self.rng.random(self.n_agents) < cfg.sleep_disruption_rate
            ) & wearable_mask
            self.sleep_delay[disruptions] = max(0, cfg.sleep_disruption_delay_steps)
        pending = self.sleep_delay > 0
        self.sleep_delay[pending] -= 1
        waking = (self.sleep_delay == 0) & pending
        self.sleep_remaining[waking] = np.int32(
            max(1, cfg.sleep_disruption_duration_steps)
        )
        active_sleep = self.sleep_remaining > 0
        sleep_delta = np.array(
            [
                cfg.sleep_disruption_hr_delta,
                cfg.sleep_disruption_hrv_delta,
                0.0,
                cfg.sleep_disruption_temperature_delta,
            ],
            dtype=np.float64,
        )
        for idx in np.flatnonzero(active_sleep):
            decay = self.sleep_remaining[idx] / max(1, cfg.sleep_disruption_duration_steps)
            add(int(idx), PerturbationCause.SLEEP_DISRUPTION, sleep_delta * decay)
        self.sleep_remaining[active_sleep] -= 1

        self.sensor_active.fill(False)
        for agent_idx in sorted(transition_indices or set()):
            if (
                0 <= agent_idx < self.n_agents
                and wearable_mask[agent_idx]
                and self.rng.random() < cfg.sensor_artifact_probability
            ):
                self.sensor_active[agent_idx] = True
        artifact_delta = np.array(
            [
                cfg.sensor_artifact_hr_delta,
                cfg.sensor_artifact_hrv_delta,
                0.0,
                cfg.sensor_artifact_temperature_delta,
            ],
            dtype=np.float64,
        )
        for idx in np.flatnonzero(self.sensor_active):
            add(int(idx), PerturbationCause.SENSOR_ARTIFACT, artifact_delta)

        heat_instance = next(
            (
                instance
                for instance in self.heat_wave_instances
                if instance.start_step <= current_step < instance.end_step
            ),
            None,
        )
        if heat_instance is not None:
            if heat_instance.instance_id != self.heat_wave_instance_id:
                self.heat_wave_amplitudes = np.maximum(
                    0.0,
                    1.0
                    + cfg.heat_wave_amplitude_jitter
                    * self.rng.normal(size=self.n_agents),
                )
                self.heat_wave_instance_id = heat_instance.instance_id
            for idx in np.flatnonzero(wearable_mask):
                amplitude = self.heat_wave_amplitudes[idx]
                heat_delta = np.array(
                    [
                        cfg.heat_wave_hr_delta * amplitude,
                        cfg.heat_wave_hrv_delta * amplitude,
                        0.0,
                        cfg.heat_wave_temperature_delta * amplitude,
                    ],
                    dtype=np.float64,
                )
                add(int(idx), PerturbationCause.HEAT_WAVE, heat_delta)
            if heat_instance.instance_id not in self.benign_instances:
                self._register_instance(
                    BenignInstance(
                        heat_instance.instance_id,
                        PerturbationCause.HEAT_WAVE,
                        heat_instance.start_step,
                        heat_instance.end_step,
                        global_scope=True,
                    )
                )
            self._active_instance_ids.add(heat_instance.instance_id)
        else:
            self.heat_wave_instance_id = None

        self._step_venue_crowding(current_step, wearable_mask, add)
        self._step_background_ili(current_step, wearable_mask, add)

        return ConfounderStep(
            contributions={idx: tuple(items) for idx, items in contributions.items()},
            affected_agents_by_cause=affected,
            heat_wave_active=heat_instance is not None,
            heat_wave_instance_id=(
                heat_instance.instance_id if heat_instance is not None else None
            ),
            heat_wave_start_step=(
                heat_instance.start_step if heat_instance is not None else None
            ),
            heat_wave_end_step=(
                heat_instance.end_step if heat_instance is not None else None
            ),
            benign_instances={
                instance_id: BenignInstance(
                    instance.instance_id,
                    instance.cause,
                    instance.start_step,
                    instance.end_step,
                    set(instance.current_agents),
                    instance.global_scope,
                )
                for instance_id in sorted(self._active_instance_ids)
                if (
                    instance := self.benign_instances.get(instance_id)
                ) is not None
            },
        )

    def _step_venue_crowding(
        self, current_step: int, wearable_mask: NDArray[np.bool_], add
    ) -> None:
        cfg = self.config
        if self.venue_engine is None or cfg.venue_crowding_rate <= 0:
            return
        allowed = set(cfg.venue_crowding_venue_types)
        for venue_idx, venue in enumerate(self.venue_engine.venues):
            if venue.venue_type not in allowed:
                continue
            active = self._venue_active.get(venue_idx)
            if active is None and self.rng.random() < cfg.venue_crowding_rate:
                sequence = self._venue_sequence.get(venue_idx, 0)
                self._venue_sequence[venue_idx] = sequence + 1
                active = BenignInstance(
                    f"venue_{venue.venue_id}_{sequence}",
                    PerturbationCause.VENUE_CROWDING,
                    current_step,
                    current_step + max(1, cfg.venue_crowding_duration_steps),
                )
                self._venue_active[venue_idx] = active
                self._register_instance(active)
                self._venue_amplitudes[active.instance_id] = np.maximum(
                    0.0,
                    1.0
                    + cfg.venue_crowding_amplitude_jitter
                    * self.rng.normal(size=self.n_agents),
                )
            if active is None:
                continue
            if current_step >= active.end_step:
                del self._venue_active[venue_idx]
                active.current_agents = set()
                continue
            present = {
                int(idx)
                for idx in self.venue_engine.agents_at_venue(venue_idx)
                if wearable_mask[idx]
            }
            active.current_agents = present
            self._active_instance_ids.add(active.instance_id)
            occupancy = len(self.venue_engine.agents_at_venue(venue_idx))
            denominator = (
                venue.capacity
                if venue.capacity is not None and venue.capacity > 0
                else cfg.venue_crowding_occupancy_reference
            )
            intensity = occupancy / max(denominator, 1.0)
            base = np.array(
                [
                    cfg.venue_crowding_hr_delta,
                    cfg.venue_crowding_hrv_delta,
                    0.0,
                    cfg.venue_crowding_temperature_delta,
                ],
                dtype=np.float64,
            )
            amplitudes = self._venue_amplitudes[active.instance_id]
            for idx in sorted(present):
                add(
                    idx,
                    PerturbationCause.VENUE_CROWDING,
                    base * intensity * amplitudes[idx],
                )

    def _step_background_ili(
        self, current_step: int, wearable_mask: NDArray[np.bool_], add
    ) -> None:
        cfg = self.config
        if (
            self.household_ids is None
            or cfg.background_ili_daily_incidence <= 0
        ):
            return
        if current_step % STEPS_PER_DAY == 0:
            candidates = np.flatnonzero(
                wearable_mask
                & (self._ili_remaining == 0)
                & (self._ili_delay == 0)
            )
            for idx in candidates:
                if self.rng.random() >= cfg.background_ili_daily_incidence:
                    continue
                instance_id = f"ili_{self._ili_sequence}"
                self._ili_sequence += 1
                members = np.flatnonzero(self.household_ids == self.household_ids[idx])
                selected = [int(idx)]
                for member in members:
                    member_int = int(member)
                    if (
                        member_int != int(idx)
                        and wearable_mask[member_int]
                        and self._ili_remaining[member_int] == 0
                        and self._ili_delay[member_int] == 0
                    ):
                        if self.rng.random() < cfg.background_ili_secondary_probability:
                            selected.append(member_int)
                instance = BenignInstance(
                    instance_id,
                    PerturbationCause.BACKGROUND_ILI,
                    current_step + cfg.background_ili_incubation_delay_steps,
                    current_step
                    + cfg.background_ili_incubation_delay_steps
                    + max(1, cfg.background_ili_symptomatic_duration_steps),
                )
                self._register_instance(instance)
                self._ili_amplitudes[instance_id] = {
                    member_int: max(
                        0.0,
                        1.0
                        + cfg.background_ili_amplitude_jitter
                        * float(self.rng.normal()),
                    )
                    for member_int in selected
                }
                for member_int in selected:
                    self._ili_delay[member_int] = max(
                        0, cfg.background_ili_incubation_delay_steps
                    )
                    self._ili_remaining[member_int] = max(
                        1, cfg.background_ili_symptomatic_duration_steps
                    )
                    self._ili_instance_by_agent[member_int] = instance_id
        pending = self._ili_delay > 0
        self._ili_delay[pending] -= 1
        active = (self._ili_delay == 0) & (self._ili_remaining > 0)
        base = np.array(
            [
                cfg.background_ili_hr_delta,
                cfg.background_ili_hrv_delta,
                0.0,
                cfg.background_ili_temperature_delta,
            ],
            dtype=np.float64,
        )
        for idx in np.flatnonzero(active & wearable_mask):
            ili_instance_id: str | None = self._ili_instance_by_agent.get(int(idx))
            if ili_instance_id is None:
                continue
            instance = self.benign_instances[ili_instance_id]
            self._active_instance_ids.add(ili_instance_id)
            instance.current_agents.add(int(idx))
            decay = self._ili_remaining[idx] / max(
                1, cfg.background_ili_symptomatic_duration_steps
            )
            amplitude = self._ili_amplitudes[ili_instance_id][int(idx)]
            add(int(idx), PerturbationCause.BACKGROUND_ILI, base * decay * amplitude)
        self._ili_remaining[active] -= 1
