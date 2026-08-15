"""Benign, cause-labelled biometric confounder sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from garland.constants import STEPS_PER_DAY
from garland.perturbations import PerturbationCause, PerturbationContribution


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


@dataclass(frozen=True)
class HeatWaveInstance:
    """Model-side identity and footprint of one heat-wave episode."""

    instance_id: str
    start_step: int
    end_step: int
    zone_ids: tuple[int, ...]


@dataclass
class ConfounderStep:
    """Per-step confounder contributions and source activity."""

    contributions: dict[int, tuple[PerturbationContribution, ...]]
    affected_agents_by_cause: dict[PerturbationCause, set[int]]
    heat_wave_active: bool = False
    heat_wave_instance_id: str | None = None
    heat_wave_start_step: int | None = None
    heat_wave_end_step: int | None = None


class ConfounderEngine:
    """Generate seeded benign perturbations independently of hazard state."""

    def __init__(
        self,
        n_agents: int,
        config: ConfoundersConfig,
        rng: np.random.Generator,
        zone_ids: tuple[int, ...] = (),
    ) -> None:
        self.n_agents = n_agents
        self.config = config
        self.rng = rng
        self.zone_ids = zone_ids
        self.exercise_remaining = np.zeros(n_agents, dtype=np.int32)
        self.sleep_delay = np.zeros(n_agents, dtype=np.int32)
        self.sleep_remaining = np.zeros(n_agents, dtype=np.int32)
        self.sensor_active = np.zeros(n_agents, dtype=bool)
        self.heat_wave_instances = self._build_heat_wave_instances()
        self.heat_wave_amplitudes = np.ones(n_agents, dtype=np.float64)
        self.heat_wave_instance_id: str | None = None

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
        else:
            self.heat_wave_instance_id = None

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
        )
