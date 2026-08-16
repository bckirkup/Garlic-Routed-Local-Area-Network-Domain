"""Benign, cause-labelled biometric confounder sources."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.channels import DEFAULT_CHANNEL_SET, ChannelSet
from garland.constants import STEPS_PER_DAY
from garland.modality_signatures import (
    contact_artifact_axes,
    exertion_axes,
    infection_axes,
    modality_delta,
)
from garland.perturbations import PerturbationCause, PerturbationContribution
from garland.venues import VenueEngine, VenueType

# Background ILI is a real, milder infection than the modelled outbreak: its
# core-vital deltas run about 0.6 of the symptomatic peak, so the band channels
# follow the same illness axes at the same fraction rather than staying quiet,
# which would make the bands a free discriminator between the two.
_BACKGROUND_ILI_SEVERITY = 0.6


@dataclass
class ConfoundersConfig:
    """Configuration for independent and structured benign event sources."""

    enabled: bool = False
    exercise_rate: float = 0.01
    exercise_duration_steps: int = 6
    exercise_hr_delta: float = 8.0
    exercise_hrv_delta: float = -6.0
    exercise_temperature_delta: float = 0.08
    sleep_disruption_rate: float = 0.05
    sleep_disruption_delay_steps: int = 96
    sleep_disruption_delay_jitter_steps: int = 24
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
    heat_wave_peak_hour: float = 15.0
    heat_wave_peak_width_hours: float = 5.0
    heat_wave_night_floor: float = 0.35
    heat_wave_ac_exposure_multiplier: float = 0.2
    heat_wave_materiality_floor: float = 0.5
    heat_wave_elderly_weight: float = 0.5
    heat_wave_outdoor_worker_weight: float = 0.75
    heat_wave_endurance_athlete_weight: float = 0.5
    elderly_fraction: float = 0.2
    has_air_conditioning_fraction: float = 0.7
    outdoor_worker_fraction: float = 0.1
    endurance_athlete_fraction: float = 0.1
    heat_island_gain: float = 0.35
    block_fire_start_step: int = 0
    block_fire_duration_steps: int = 0
    block_fire_center_x: float = 0.0
    block_fire_center_y: float = 0.0
    block_fire_radius_m: float = 100.0
    block_fire_materiality_floor: float = 0.25
    block_fire_elderly_weight: float = 0.5
    block_fire_hr_delta: float = 5.0
    block_fire_hrv_delta: float = -4.0
    block_fire_respiratory_delta: float = 4.0
    block_fire_temperature_delta: float = 0.0
    block_fire_amplitude_jitter: float = 0.1
    victory_start_step: int = 0
    victory_duration_steps: int = 0
    victory_fan_fraction: float = 0.25
    victory_participation_fraction: float = 0.8
    victory_onset_jitter_steps: int = 3
    victory_hr_delta: float = 6.0
    victory_hrv_delta: float = -5.0
    victory_temperature_delta: float = 0.05
    victory_amplitude_jitter: float = 0.1
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

    def __post_init__(self) -> None:
        fractions = {
            "elderly_fraction": self.elderly_fraction,
            "has_air_conditioning_fraction": self.has_air_conditioning_fraction,
            "outdoor_worker_fraction": self.outdoor_worker_fraction,
            "endurance_athlete_fraction": self.endurance_athlete_fraction,
            "victory_fan_fraction": self.victory_fan_fraction,
            "victory_participation_fraction": self.victory_participation_fraction,
        }
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.heat_wave_peak_width_hours <= 0.0:
            raise ValueError("heat_wave_peak_width_hours must be positive")
        if self.heat_wave_night_floor < 0.0:
            raise ValueError("heat_wave_night_floor must be non-negative")
        if self.heat_wave_ac_exposure_multiplier < 0.0:
            raise ValueError("heat_wave_ac_exposure_multiplier must be non-negative")
        if self.heat_wave_materiality_floor < 0.0:
            raise ValueError("heat_wave_materiality_floor must be non-negative")
        if self.heat_island_gain < 0.0:
            raise ValueError("heat_island_gain must be non-negative")
        if self.sleep_disruption_delay_jitter_steps < 0:
            raise ValueError("sleep_disruption_delay_jitter_steps must be non-negative")
        nonnegative_values = {
            "block_fire_start_step": self.block_fire_start_step,
            "block_fire_duration_steps": self.block_fire_duration_steps,
            "victory_start_step": self.victory_start_step,
            "victory_duration_steps": self.victory_duration_steps,
            "victory_onset_jitter_steps": self.victory_onset_jitter_steps,
            "block_fire_materiality_floor": self.block_fire_materiality_floor,
            "block_fire_elderly_weight": self.block_fire_elderly_weight,
            "block_fire_amplitude_jitter": self.block_fire_amplitude_jitter,
            "victory_amplitude_jitter": self.victory_amplitude_jitter,
        }
        for name, value in nonnegative_values.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.block_fire_radius_m <= 0.0:
            raise ValueError("block_fire_radius_m must be positive")


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
        agent_x: NDArray[np.float64] | None = None,
        agent_y: NDArray[np.float64] | None = None,
        exposure_rng: np.random.Generator | None = None,
        channel_set: ChannelSet = DEFAULT_CHANNEL_SET,
    ) -> None:
        self.n_agents = n_agents
        self.config = config
        self.channel_set = channel_set
        self.rng = rng
        self.zone_ids = zone_ids
        self.household_ids = household_ids
        self.venue_engine = venue_engine
        self.initial_agent_x = (
            np.asarray(agent_x, dtype=np.float64).copy()
            if agent_x is not None
            else np.zeros(n_agents, dtype=np.float64)
        )
        self.initial_agent_y = (
            np.asarray(agent_y, dtype=np.float64).copy()
            if agent_y is not None
            else np.zeros(n_agents, dtype=np.float64)
        )
        self.elderly = np.zeros(n_agents, dtype=bool)
        self.has_air_conditioning = np.zeros(n_agents, dtype=bool)
        self.outdoor_worker = np.zeros(n_agents, dtype=bool)
        self.endurance_athlete = np.zeros(n_agents, dtype=bool)
        self.sports_fan = np.zeros(n_agents, dtype=bool)
        self.heat_island_factor = np.ones(n_agents, dtype=np.float64)
        if config.enabled:
            exposure_rng = exposure_rng or np.random.default_rng(
                np.random.SeedSequence([0xE5, n_agents])
            )
            self.elderly = exposure_rng.random(n_agents) < config.elderly_fraction
            self.has_air_conditioning = (
                exposure_rng.random(n_agents) < config.has_air_conditioning_fraction
            )
            self.outdoor_worker = exposure_rng.random(n_agents) < config.outdoor_worker_fraction
            self.endurance_athlete = (
                exposure_rng.random(n_agents) < config.endurance_athlete_fraction
            )
            self.sports_fan = exposure_rng.random(n_agents) < config.victory_fan_fraction
            if agent_x is not None and agent_y is not None:
                center_x = float(np.mean(agent_x))
                center_y = float(np.mean(agent_y))
                distance = np.hypot(agent_x - center_x, agent_y - center_y)
                max_distance = float(np.max(distance))
                core_fraction = (
                    1.0 - distance / max_distance if max_distance > 0.0 else np.ones(n_agents)
                )
                self.heat_island_factor = np.clip(
                    1.0 + config.heat_island_gain * core_fraction,
                    0.0,
                    1.0 + max(config.heat_island_gain, 0.0),
                )
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
        self.block_fire_instance_id: str | None = None
        self.block_fire_amplitudes = np.ones(n_agents, dtype=np.float64)
        self.victory_instance_id: str | None = None
        self.victory_participating = np.zeros(n_agents, dtype=bool)
        self.victory_onset_steps = np.full(n_agents, -1, dtype=np.int32)
        self.victory_amplitudes = np.ones(n_agents, dtype=np.float64)

    def _register_instance(self, instance: BenignInstance) -> None:
        self.benign_instances[instance.instance_id] = instance
        heapq.heappush(self._instance_expiry, (instance.end_step, instance.instance_id))

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

    def _block_fire_distance_weights(
        self,
        agent_x: NDArray[np.float64],
        agent_y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        cfg = self.config
        distance = np.hypot(
            agent_x - cfg.block_fire_center_x,
            agent_y - cfg.block_fire_center_y,
        )
        return np.where(
            distance <= 3.0 * cfg.block_fire_radius_m,
            np.exp(-0.5 * (distance / cfg.block_fire_radius_m) ** 2),
            0.0,
        )

    def _step_block_fire(
        self,
        current_step: int,
        wearable_mask: NDArray[np.bool_],
        agent_x: NDArray[np.float64],
        agent_y: NDArray[np.float64],
        add,
    ) -> None:
        cfg = self.config
        if not (
            cfg.block_fire_start_step
            <= current_step
            < cfg.block_fire_start_step + cfg.block_fire_duration_steps
        ):
            self.block_fire_instance_id = None
            return
        if self.block_fire_instance_id is None:
            self.block_fire_instance_id = "block_fire_0"
            self.block_fire_amplitudes = np.maximum(
                0.0,
                1.0 + cfg.block_fire_amplitude_jitter * self.rng.normal(size=self.n_agents),
            )
        distance_weight = self._block_fire_distance_weights(agent_x, agent_y)
        susceptibility = 1.0 + cfg.block_fire_elderly_weight * self.elderly
        weights = distance_weight * susceptibility * self.block_fire_amplitudes
        material = wearable_mask & (weights >= cfg.block_fire_materiality_floor)
        instance = self.benign_instances.get(self.block_fire_instance_id)
        if instance is None:
            instance = BenignInstance(
                self.block_fire_instance_id,
                PerturbationCause.IRRITANT_EXPOSURE,
                cfg.block_fire_start_step,
                cfg.block_fire_start_step + cfg.block_fire_duration_steps,
                set(np.flatnonzero(material).tolist()),
            )
            self._register_instance(instance)
        else:
            instance.current_agents = set(np.flatnonzero(material).tolist())
        self._active_instance_ids.add(self.block_fire_instance_id)
        base = self.channel_set.delta(
            {
                "heart_rate": cfg.block_fire_hr_delta,
                "hrv_rmssd": cfg.block_fire_hrv_delta,
                "respiratory_rate": cfg.block_fire_respiratory_delta,
                "body_temperature": cfg.block_fire_temperature_delta,
            }
        )
        for idx in np.flatnonzero(wearable_mask & (weights > 1e-9)):
            add(
                int(idx),
                PerturbationCause.IRRITANT_EXPOSURE,
                base * weights[idx],
            )

    def _step_victory(
        self,
        current_step: int,
        wearable_mask: NDArray[np.bool_],
        add,
    ) -> None:
        cfg = self.config
        if cfg.victory_duration_steps <= 0:
            return
        event_end = sum(
            (
                cfg.victory_start_step,
                cfg.victory_duration_steps,
                cfg.victory_onset_jitter_steps,
            )
        )
        if not cfg.victory_start_step <= current_step < event_end:
            return
        if self.victory_instance_id is None:
            self.victory_instance_id = "victory_0"
            self.victory_participating = self.sports_fan & (
                self.rng.random(self.n_agents) < cfg.victory_participation_fraction
            )
            self.victory_onset_steps.fill(-1)
            jitter = self.rng.integers(
                0,
                cfg.victory_onset_jitter_steps + 1,
                size=self.n_agents,
            )
            self.victory_onset_steps[self.victory_participating] = (
                cfg.victory_start_step + jitter[self.victory_participating]
            )
            self.victory_amplitudes = np.maximum(
                0.0,
                1.0 + cfg.victory_amplitude_jitter * self.rng.normal(size=self.n_agents),
            )
            self._register_instance(
                BenignInstance(
                    self.victory_instance_id,
                    PerturbationCause.SLEEP_DISRUPTION,
                    cfg.victory_start_step,
                    event_end,
                )
            )
        active = (
            self.victory_participating
            & wearable_mask
            & (current_step >= self.victory_onset_steps)
            & (current_step < self.victory_onset_steps + cfg.victory_duration_steps)
        )
        instance = self.benign_instances[self.victory_instance_id]
        instance.current_agents = set(np.flatnonzero(active).tolist())
        self._active_instance_ids.add(self.victory_instance_id)
        base = self.channel_set.delta(
            {
                "heart_rate": cfg.victory_hr_delta,
                "hrv_rmssd": cfg.victory_hrv_delta,
                "body_temperature": cfg.victory_temperature_delta,
            }
        )
        for idx in np.flatnonzero(active):
            elapsed = current_step - self.victory_onset_steps[idx]
            decay = max(
                0.0,
                (cfg.victory_duration_steps - elapsed) / max(cfg.victory_duration_steps, 1),
            )
            add(
                int(idx),
                PerturbationCause.SLEEP_DISRUPTION,
                base * decay * self.victory_amplitudes[idx],
            )

    def step(
        self,
        current_step: int,
        hour_of_day: float,
        wearable_mask: NDArray[np.bool_],
        transition_indices: set[int] | None = None,
        agent_x: NDArray[np.float64] | None = None,
        agent_y: NDArray[np.float64] | None = None,
    ) -> ConfounderStep:
        """Advance sources and return this step's labelled contributions."""
        if not self.config.enabled:
            return ConfounderStep({}, {})

        cfg = self.config
        current_x = self.initial_agent_x if agent_x is None else agent_x
        current_y = self.initial_agent_y if agent_y is None else agent_y
        contributions: dict[int, list[PerturbationContribution]] = {}
        affected: dict[PerturbationCause, set[int]] = {}
        self._prune_instances(current_step)
        for instance_id in self._active_instance_ids:
            instance = self.benign_instances.get(instance_id)
            if instance is not None and instance.cause == PerturbationCause.BACKGROUND_ILI:
                instance.current_agents.clear()
        self._active_instance_ids.clear()

        def add(
            agent_idx: int,
            cause: PerturbationCause,
            delta: NDArray[np.float64],
        ) -> None:
            if not wearable_mask[agent_idx]:
                return
            contributions.setdefault(agent_idx, []).append(PerturbationContribution(cause, delta))
            affected.setdefault(cause, set()).add(agent_idx)

        active_exercise = self.exercise_remaining > 0
        exercise_weight = (
            0.25 + 0.75 * max(0.0, np.sin(np.pi * (hour_of_day - 6.0) / 12.0))
            if 6.0 <= hour_of_day <= 18.0
            else 0.1
        )
        exercise_onsets = self.rng.random(self.n_agents) < cfg.exercise_rate * exercise_weight
        exercise_onsets &= ~active_exercise & wearable_mask
        self.exercise_remaining[exercise_onsets] = max(1, cfg.exercise_duration_steps)
        active_exercise = self.exercise_remaining > 0
        exercise_delta = self.channel_set.delta(
            {
                "heart_rate": cfg.exercise_hr_delta,
                "hrv_rmssd": cfg.exercise_hrv_delta,
                "body_temperature": cfg.exercise_temperature_delta,
            }
        ) + modality_delta(exertion_axes(1.0), self.channel_set)
        for idx in np.flatnonzero(active_exercise):
            add(int(idx), PerturbationCause.EXERCISE, exercise_delta)
        self.exercise_remaining[active_exercise] -= 1

        nightly_step = current_step % STEPS_PER_DAY == 22 * 12
        if nightly_step:
            disruptions = (
                self.rng.random(self.n_agents) < cfg.sleep_disruption_rate
            ) & wearable_mask
            self.sleep_delay[disruptions] = max(0, cfg.sleep_disruption_delay_steps)
            jitter = max(0, cfg.sleep_disruption_delay_jitter_steps)
            if jitter:
                offsets = self.rng.integers(-jitter, jitter + 1, size=self.n_agents)
                self.sleep_delay[disruptions] = np.maximum(
                    0,
                    self.sleep_delay[disruptions] + offsets[disruptions],
                )
        pending = self.sleep_delay > 0
        self.sleep_delay[pending] -= 1
        waking = (self.sleep_delay == 0) & pending
        self.sleep_remaining[waking] = np.int32(max(1, cfg.sleep_disruption_duration_steps))
        active_sleep = self.sleep_remaining > 0
        sleep_delta = self.channel_set.delta(
            {
                "heart_rate": cfg.sleep_disruption_hr_delta,
                "hrv_rmssd": cfg.sleep_disruption_hrv_delta,
                "body_temperature": cfg.sleep_disruption_temperature_delta,
            }
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
        artifact_delta = self.channel_set.delta(
            {
                "heart_rate": cfg.sensor_artifact_hr_delta,
                "hrv_rmssd": cfg.sensor_artifact_hrv_delta,
                "body_temperature": cfg.sensor_artifact_temperature_delta,
            }
        ) + modality_delta(contact_artifact_axes(1.0), self.channel_set)
        for idx in np.flatnonzero(self.sensor_active):
            add(int(idx), PerturbationCause.SENSOR_ARTIFACT, artifact_delta)

        heat_instance = self._step_heat_wave(current_step, hour_of_day, wearable_mask, add)

        self._step_block_fire(current_step, wearable_mask, current_x, current_y, add)
        self._step_victory(current_step, wearable_mask, add)
        self._step_venue_crowding(current_step, wearable_mask, add)
        self._step_background_ili(current_step, wearable_mask, add)

        return ConfounderStep(
            contributions={idx: tuple(items) for idx, items in contributions.items()},
            affected_agents_by_cause=affected,
            heat_wave_active=heat_instance is not None,
            heat_wave_instance_id=(
                heat_instance.instance_id if heat_instance is not None else None
            ),
            heat_wave_start_step=(heat_instance.start_step if heat_instance is not None else None),
            heat_wave_end_step=(heat_instance.end_step if heat_instance is not None else None),
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
                if (instance := self.benign_instances.get(instance_id)) is not None
            },
        )

    def _step_heat_wave(
        self,
        current_step: int,
        hour_of_day: float,
        wearable_mask: NDArray[np.bool_],
        add,
    ) -> HeatWaveInstance | None:
        cfg = self.config
        heat_instance = next(
            (
                instance
                for instance in self.heat_wave_instances
                if instance.start_step <= current_step < instance.end_step
            ),
            None,
        )
        if heat_instance is None:
            self.heat_wave_instance_id = None
            return None
        if heat_instance.instance_id != self.heat_wave_instance_id:
            self.heat_wave_amplitudes = np.maximum(
                0.0,
                1.0 + cfg.heat_wave_amplitude_jitter * self.rng.normal(size=self.n_agents),
            )
            self.heat_wave_instance_id = heat_instance.instance_id
        weights, heat_affected = self._heat_wave_weights(hour_of_day, wearable_mask)
        material_heat_affected = heat_affected & (weights >= cfg.heat_wave_materiality_floor)
        affected_indices = {int(idx) for idx in np.flatnonzero(material_heat_affected)}
        heat_base = self.channel_set.delta(
            {
                "heart_rate": cfg.heat_wave_hr_delta,
                "hrv_rmssd": cfg.heat_wave_hrv_delta,
                "body_temperature": cfg.heat_wave_temperature_delta,
            }
        )
        for idx in np.flatnonzero(heat_affected):
            amplitude = self.heat_wave_amplitudes[idx]
            weight = weights[idx]
            add(int(idx), PerturbationCause.HEAT_WAVE, heat_base * weight * amplitude)
        self._update_heat_instance(heat_instance, affected_indices)
        self._active_instance_ids.add(heat_instance.instance_id)
        return heat_instance

    def _heat_wave_weights(
        self, hour_of_day: float, wearable_mask: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        cfg = self.config
        hours_from_peak = (hour_of_day - cfg.heat_wave_peak_hour + 12.0) % 24.0 - 12.0
        diurnal = float(
            np.exp(-0.5 * (hours_from_peak / max(cfg.heat_wave_peak_width_hours, 0.1)) ** 2)
        )
        exposure = (
            1.0
            + cfg.heat_wave_elderly_weight * self.elderly
            + cfg.heat_wave_outdoor_worker_weight * self.outdoor_worker
            + cfg.heat_wave_endurance_athlete_weight * self.endurance_athlete
        )
        exposure *= self.heat_island_factor
        intensity = np.where(
            self.has_air_conditioning,
            diurnal * cfg.heat_wave_ac_exposure_multiplier,
            np.maximum(diurnal, cfg.heat_wave_night_floor),
        )
        weights = exposure * intensity
        return weights, wearable_mask & (weights > 1e-9)

    def _update_heat_instance(
        self, heat_instance: HeatWaveInstance, affected_agents: set[int]
    ) -> None:
        if heat_instance.instance_id not in self.benign_instances:
            self._register_instance(
                BenignInstance(
                    heat_instance.instance_id,
                    PerturbationCause.HEAT_WAVE,
                    heat_instance.start_step,
                    heat_instance.end_step,
                    current_agents=affected_agents,
                    global_scope=False,
                )
            )
            return
        self.benign_instances[heat_instance.instance_id].current_agents = affected_agents

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
                    1.0 + cfg.venue_crowding_amplitude_jitter * self.rng.normal(size=self.n_agents),
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
            base = self.channel_set.delta(
                {
                    "heart_rate": cfg.venue_crowding_hr_delta,
                    "hrv_rmssd": cfg.venue_crowding_hrv_delta,
                    "body_temperature": cfg.venue_crowding_temperature_delta,
                }
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
        if self.household_ids is None or cfg.background_ili_daily_incidence <= 0:
            return
        if current_step % STEPS_PER_DAY == 0:
            candidates = np.flatnonzero(
                wearable_mask & (self._ili_remaining == 0) & (self._ili_delay == 0)
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
                        1.0 + cfg.background_ili_amplitude_jitter * float(self.rng.normal()),
                    )
                    for member_int in selected
                }
                for member_int in selected:
                    self._ili_delay[member_int] = max(0, cfg.background_ili_incubation_delay_steps)
                    self._ili_remaining[member_int] = max(
                        1, cfg.background_ili_symptomatic_duration_steps
                    )
                    self._ili_instance_by_agent[member_int] = instance_id
        pending = self._ili_delay > 0
        self._ili_delay[pending] -= 1
        active = (self._ili_delay == 0) & (self._ili_remaining > 0)
        base = self.channel_set.delta(
            {
                "heart_rate": cfg.background_ili_hr_delta,
                "hrv_rmssd": cfg.background_ili_hrv_delta,
                "body_temperature": cfg.background_ili_temperature_delta,
            }
        ) + modality_delta(infection_axes(_BACKGROUND_ILI_SEVERITY), self.channel_set)
        for idx in np.flatnonzero(active & wearable_mask):
            ili_instance_id: str | None = self._ili_instance_by_agent.get(int(idx))
            if ili_instance_id is None:
                continue
            instance = self.benign_instances[ili_instance_id]
            self._active_instance_ids.add(ili_instance_id)
            instance.current_agents.add(int(idx))
            decay = self._ili_remaining[idx] / max(1, cfg.background_ili_symptomatic_duration_steps)
            amplitude = self._ili_amplitudes[ili_instance_id][int(idx)]
            add(int(idx), PerturbationCause.BACKGROUND_ILI, base * decay * amplitude)
        self._ili_remaining[active] -= 1
