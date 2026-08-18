"""Evaluation metrics for the GARLAND epidemiological security testbed.

Tracks and outputs:
- Time-to-Detection for both hazard types (disease and toxin)
- False Positive / False Negative rates
- Hazard discrimination (decoupling toxin from infection)
- Cumulative Privacy Budget (Epsilon) under adaptive composition
- Per-step CSV output for downstream analysis
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from garland.constants import STEPS_PER_DAY
from garland.detection_power import DetectionPowerTracker
from garland.paths import (
    ensure_directory,
    resolve_under_base,
    save_figure,
    write_text_file,
)
from garland.perturbations import BENIGN_CAUSES, PerturbationCause
from garland.privacy import AnomalyType

_TIME_HOURS_LABEL = "Time (hours)"


@dataclass
class _HazardEpisodeState:
    """Tracks hazard and no-hazard episodes for episode-granular FN/TN counting."""

    hazard_active: bool = False
    detected_in_hazard_episode: bool = False
    false_positive_in_no_hazard_episode: bool = False


@dataclass
class DetectionEvent:
    """Record of a hazard detection by the system."""

    step: int
    hazard_type: str  # "disease" or "toxin"
    anomaly_type: AnomalyType
    zone_id: int
    true_positive: bool
    agents_affected: int
    dosed_agents: int | None = None
    hazard_instance_id: str | None = None
    attributed: bool | None = None
    causes: frozenset[PerturbationCause] = frozenset()
    benign_instance_id: str | None = None
    benign_attributed: bool = False
    benign_cause: PerturbationCause | None = None
    benign_causes: frozenset[PerturbationCause] = frozenset()


class WarrantClass(str, Enum):
    """Reporting class for what a detection warrants."""

    TARGET = "target"
    ACTIONABLE_NON_TARGET = "actionable_non_target"
    EXPLAINED = "explained"
    ARTIFACT = "artifact"
    UNEXPLAINED = "unexplained"


@dataclass
class _BackgroundFoldState:
    open_time_bin: int | None = None
    open_groups: dict[tuple[int, AnomalyType], list[int]] = field(default_factory=dict)
    emission_n: dict[AnomalyType, int] = field(default_factory=dict)
    emission_sum_c: dict[AnomalyType, int] = field(default_factory=dict)
    emission_sum_e: dict[AnomalyType, int] = field(default_factory=dict)
    emission_sum_c2_over_e: dict[AnomalyType, float] = field(default_factory=dict)
    emission_sum_c2: dict[AnomalyType, int] = field(default_factory=dict)
    emission_e_hist: dict[AnomalyType, dict[int, int]] = field(default_factory=dict)
    emission_bucket_stats: dict[tuple[AnomalyType, str], list[float]] = field(default_factory=dict)
    emission_observed: dict[AnomalyType, int] = field(default_factory=dict)
    window_history: dict[tuple[int, AnomalyType], deque[tuple[int, int]]] = field(
        default_factory=dict
    )
    window_n: dict[AnomalyType, int] = field(default_factory=dict)
    window_sum_c: dict[AnomalyType, int] = field(default_factory=dict)
    window_sum_e: dict[AnomalyType, int] = field(default_factory=dict)
    window_sum_c2_over_e: dict[AnomalyType, float] = field(default_factory=dict)
    window_sum_c2: dict[AnomalyType, int] = field(default_factory=dict)
    window_e_hist: dict[AnomalyType, dict[int, int]] = field(default_factory=dict)
    window_bucket_stats: dict[tuple[AnomalyType, str], list[float]] = field(default_factory=dict)
    window_observed: dict[AnomalyType, int] = field(default_factory=dict)


@dataclass
class MetricsCollector:
    """Collects and computes evaluation metrics throughout the simulation.

    Attributes
    ----------
    disease_onset_step : int | None
        Step when first infection became detectable (E→I transition).
    toxin_onset_step : int | None
        Step when plume started.
    disease_detection_step : int | None
        Step when system first detected disease cluster.
    toxin_detection_step : int | None
        Step when system first detected toxin exposure.
    """

    # Ground truth onset times
    disease_onset_step: int | None = None
    toxin_onset_step: int | None = None

    # Per-hazard-instance tracking (multi-plume / multi-outbreak)
    disease_onset_steps: dict[str, int] = field(default_factory=dict)
    toxin_onset_steps: dict[str, int] = field(default_factory=dict)
    instance_true_positives: dict[str, int] = field(default_factory=dict)

    # Device-local baseline maturation (evaluation-only)
    baseline_maturation_minimum_history_days: int = 0
    baseline_maturation_maximum_history_days: int = 0
    baseline_maturation_cadence_steps: int = 1
    baseline_maturation_device_count: int = 0
    baseline_maturation_history_days: list[int] = field(default_factory=list)
    baseline_maturation_sample_counts: list[int] = field(default_factory=list)
    baseline_maturation_circadian_bins: list[int] = field(default_factory=list)
    baseline_maturation_monthly_bins: list[int] = field(default_factory=list)

    # System detection times
    disease_detection_step: int | None = None
    toxin_detection_step: int | None = None

    # Per-step tracking
    step_records: list[dict] = field(default_factory=list)
    detection_events: list[DetectionEvent] = field(default_factory=list)
    dilation_records: list[dict[str, int | float | bool]] = field(default_factory=list)
    dilation_suppressed_records: list[dict[str, int]] = field(default_factory=list)
    dilation_release_suppressed_records: list[dict[str, int | float]] = field(default_factory=list)

    # Confusion matrix accumulators
    true_positives_disease: int = 0
    false_positives_disease: int = 0
    true_negatives_disease: int = 0
    false_negatives_disease: int = 0
    true_positives_toxin: int = 0
    toxin_true_positives_fewer_than_two_dosed_agents: int = 0
    false_positives_toxin: int = 0
    true_negatives_toxin: int = 0
    false_negatives_toxin: int = 0
    attributed_true_positives_disease: int = 0
    attributed_true_positives_toxin: int = 0
    coincidental_true_positives_disease: int = 0
    coincidental_true_positives_toxin: int = 0
    attributed_disease_detection_step: int | None = None
    attributed_toxin_detection_step: int | None = None
    affected_agent_token_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"disease": {}, "toxin": {}}
    )
    largest_affected_agent_group: dict[str, int] = field(
        default_factory=lambda: {"disease": 0, "toxin": 0}
    )

    # Privacy tracking
    epsilon_per_step: list[float] = field(default_factory=list)
    total_queries_issued: int = 0
    total_responses: int = 0
    disambiguation_queries_issued: int = 0
    disambiguation_asks_suppressed_by_budget: int = 0
    disambiguation_acks: int = 0
    disambiguation_ack_release_count: int = 0
    disambiguation_devices_reached: int = 0
    disambiguation_yes_answers: int = 0
    disambiguation_no_answers: int = 0
    disambiguation_unanswered_expired: int = 0
    disambiguation_unresolved_hypotheses: int = 0
    disambiguation_answer_epsilon: float = 0.0
    disambiguation_ack_epsilon: float = 0.0
    disambiguation_well_founded_queries: int = 0
    disambiguation_unfounded_queries: int = 0
    disambiguation_unscored_queries: int = 0
    disambiguation_unfounded_ask_epsilon: float = 0.0
    disambiguation_unscored_ask_epsilon: float = 0.0
    disambiguation_max_ask_epsilon_delta: float = 0.0
    aggregate_count_releases: list[dict[str, int | float | str]] = field(default_factory=list)
    aggregate_count_no_cluster_releases: int = 0
    aggregate_count_below_floor_releases: int = 0
    aggregate_count_evidence_releases: int = 0
    response_mechanism: str = "aggregate_noisy_count"
    aggregate_count_epsilon_per_release: float = 0.0
    aggregate_count_epsilon: float = 0.0
    aggregate_count_composed_epsilon: float = 0.0
    aggregate_count_evidence_threshold: int = 0
    aggregate_count_minimum_releasable_count: int = 0
    response_epsilon_basis: str = "mechanism"
    response_epsilon_per_response: float = 0.0
    unaffected_positive_reply_probability: float | None = None
    geo_epsilon_per_metre: float = 0.0
    geo_epsilon_basis: str = "separate"
    disambiguation_ack_epsilon_basis: str = "configured"
    disambiguation_well_founded_by_hypothesis: dict[str, int] = field(default_factory=dict)
    disambiguation_unfounded_by_hypothesis: dict[str, int] = field(default_factory=dict)
    disambiguation_unscored_by_hypothesis: dict[str, int] = field(default_factory=dict)
    confounder_contributions_by_cause: dict[str, int] = field(default_factory=dict)
    confounder_agents_affected_by_cause: dict[str, set[int]] = field(default_factory=dict)
    heat_wave_active_steps: int = 0
    heat_wave_instances: dict[str, dict[str, object]] = field(default_factory=dict)
    benign_overlap_detections: int = 0
    benign_attributed_detections: int = 0
    benign_misattributed_detections: int = 0
    benign_misattributions_by_cause: dict[str, int] = field(default_factory=dict)
    benign_coincident_true_positives: int = 0
    target_detections: int = 0
    actionable_non_target_detections: int = 0
    explained_detections: int = 0
    artifact_detections: int = 0
    unexplained_detections: int = 0
    warranted_detections: int = 0

    # Attack metrics
    sybil_false_alerts: int = 0
    deanon_attempts: int = 0
    deanon_successes: int = 0
    eclipse_tokens_dropped: int = 0
    correlation_evaluations: int = 0
    correlation_successes: int = 0
    replay_tokens_injected: int = 0
    replay_false_alerts: int = 0

    # Warm-up tracking
    baseline_warmup_steps: int = 0
    n_agents: int = 0
    anomaly_threshold: float | None = None
    aggregation_threshold: int | None = None
    aggregation_window_bins: int = 1
    detector_mode: str | None = None
    sequential_reference_value: float | None = None
    sequential_threshold: float | None = None
    sequential_clear_steps: int | None = None
    sequential_clear_fraction: float | None = None
    sequential_residual_ewma_alpha: float | None = None
    _daily_occupied_zones: dict[int, set[int]] = field(default_factory=dict)
    _daily_alarming_zones: dict[int, set[int]] = field(default_factory=dict)
    _background_daily_tokens: dict[int, int] = field(default_factory=dict)
    _background_daily_eligible: dict[int, int] = field(default_factory=dict)
    _background_type_tokens: dict[AnomalyType, int] = field(default_factory=dict)
    _background_type_eligible: dict[AnomalyType, int] = field(default_factory=dict)
    _background_total_tokens: int = 0
    _background_total_eligible: int = 0
    _background_population_n: int = 0
    _background_population_sum: int = 0
    _background_population_sum_sq: int = 0
    _background_agent_tokens: list[int] = field(default_factory=list)
    _background_open_time_bin: int | None = None
    _background_open_groups: dict[tuple[int, AnomalyType], list[int]] = field(default_factory=dict)
    _background_emission_n: dict[AnomalyType, int] = field(default_factory=dict)
    _background_emission_sum_c: dict[AnomalyType, int] = field(default_factory=dict)
    _background_emission_sum_e: dict[AnomalyType, int] = field(default_factory=dict)
    _background_emission_sum_c2_over_e: dict[AnomalyType, float] = field(default_factory=dict)
    _background_emission_sum_c2: dict[AnomalyType, int] = field(default_factory=dict)
    _background_emission_e_hist: dict[AnomalyType, dict[int, int]] = field(default_factory=dict)
    _background_emission_bucket_stats: dict[tuple[AnomalyType, str], list[float]] = field(
        default_factory=dict
    )
    _background_emission_observed: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_history: dict[tuple[int, AnomalyType], deque[tuple[int, int]]] = field(
        default_factory=dict
    )
    _background_window_sum_c: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_sum_e: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_sum_c2_over_e: dict[AnomalyType, float] = field(default_factory=dict)
    _background_window_sum_c2: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_sum_c_fold: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_sum_e_fold: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_n: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_e_hist: dict[AnomalyType, dict[int, int]] = field(default_factory=dict)
    _background_window_bucket_stats: dict[tuple[AnomalyType, str], list[float]] = field(
        default_factory=dict
    )
    _background_window_observed: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_triggered: dict[tuple[int, AnomalyType], bool] = field(default_factory=dict)
    world_settling_steps: int = 0
    _background_settled_tokens: int = 0
    _background_settled_eligible: int = 0
    _background_settled_population_n: int = 0
    _background_settled_population_sum: int = 0
    _background_settled_population_sum_sq: int = 0
    _background_settled_type_tokens: dict[AnomalyType, int] = field(default_factory=dict)
    _background_settled_type_eligible: dict[AnomalyType, int] = field(default_factory=dict)
    _background_settled: _BackgroundFoldState = field(default_factory=_BackgroundFoldState)
    _background_assessment_finalized: dict[str, object] | None = None
    _wearable_steps: int = 0
    _cold_baseline_wearable_steps: int = 0
    _post_world_settling_wearable_steps: int = 0
    _post_world_settling_cold_baseline_wearable_steps: int = 0
    # True when a wearable emitted while n_samples < 5 covariance prior applied.
    fleet_cold_start: bool = False
    device_re_adoption_count: int = 0
    legacy_device_adoption_warmup_reset_count: int = 0
    adoption_events: list[dict[str, int]] = field(default_factory=list)
    peak_onboarding_cold_wearables_in_zone: int = 0
    peak_onboarding_wearables_in_zone: int = 0
    # Agent-epoch detection outcomes keyed by effective width and device kind.
    detection_power: DetectionPowerTracker = field(default_factory=DetectionPowerTracker)

    def record_background_step(
        self,
        step: int,
        time_bin: int,
        eligible_by_zone: dict[int, int],
        background_by_group: dict[tuple[int, AnomalyType], int],
        background_by_agent: dict[int, int] | None = None,
    ) -> None:
        """Record model-side background emissions without entering the protocol."""
        if (
            self._background_open_time_bin is not None
            and time_bin != self._background_open_time_bin
        ):
            self._finalize_background_bin()
        if self._background_open_time_bin is None:
            self._background_open_time_bin = time_bin
        day = step // STEPS_PER_DAY
        total_eligible = sum(eligible_by_zone.values())
        total_background = sum(background_by_group.values())
        self._background_total_tokens += total_background
        self._background_total_eligible += total_eligible
        self._background_population_n += 1
        self._background_population_sum += total_background
        self._background_population_sum_sq += total_background * total_background
        if not self._background_agent_tokens and self.n_agents:
            self._background_agent_tokens = [0] * self.n_agents
        for agent_id, count in (background_by_agent or {}).items():
            if agent_id < len(self._background_agent_tokens):
                self._background_agent_tokens[agent_id] += count
        self._background_daily_tokens[day] = (
            self._background_daily_tokens.get(day, 0) + total_background
        )
        self._background_daily_eligible[day] = (
            self._background_daily_eligible.get(day, 0) + total_eligible
        )
        for (zone_id, anomaly_type), count in background_by_group.items():
            self._background_type_tokens[anomaly_type] = (
                self._background_type_tokens.get(anomaly_type, 0) + count
            )
        for zone_id, eligible in eligible_by_zone.items():
            for anomaly_type in AnomalyType:
                key = (zone_id, anomaly_type)
                counts = self._background_open_groups.setdefault(key, [0, 0])
                counts[1] += eligible
                self._background_type_eligible[anomaly_type] = (
                    self._background_type_eligible.get(anomaly_type, 0) + eligible
                )
        for key, count in background_by_group.items():
            self._background_open_groups.setdefault(key, [0, 0])[0] += count
        if step >= self.world_settling_steps:
            self._record_settled_background_step(
                step,
                time_bin,
                eligible_by_zone,
                background_by_group,
            )

    def _record_settled_background_step(
        self,
        step: int,
        time_bin: int,
        eligible_by_zone: dict[int, int],
        background_by_group: dict[tuple[int, AnomalyType], int],
    ) -> None:
        state = self._background_settled
        if state.open_time_bin is not None and time_bin != state.open_time_bin:
            self._finalize_background_state(state)
        if state.open_time_bin is None:
            state.open_time_bin = time_bin
        total_eligible = sum(eligible_by_zone.values())
        total_background = sum(background_by_group.values())
        self._background_settled_tokens += total_background
        self._background_settled_eligible += total_eligible
        self._background_settled_population_n += 1
        self._background_settled_population_sum += total_background
        self._background_settled_population_sum_sq += total_background * total_background
        for (_zone_id, anomaly_type), count in background_by_group.items():
            self._background_settled_type_tokens[anomaly_type] = (
                self._background_settled_type_tokens.get(anomaly_type, 0) + count
            )
        for eligible in eligible_by_zone.values():
            for anomaly_type in AnomalyType:
                self._background_settled_type_eligible[anomaly_type] = (
                    self._background_settled_type_eligible.get(anomaly_type, 0) + eligible
                )
        for zone_id, eligible in eligible_by_zone.items():
            for anomaly_type in AnomalyType:
                key = (zone_id, anomaly_type)
                counts = state.open_groups.setdefault(key, [0, 0])
                counts[1] += eligible
        for key, count in background_by_group.items():
            state.open_groups.setdefault(key, [0, 0])[0] += count

    @staticmethod
    def _occupancy_bucket(eligible: int) -> str:
        edges = (0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, math.inf)
        lower = max(edge for edge in edges if eligible >= edge)
        upper = next(edge for edge in edges if edge > lower)
        return f"{int(lower)}-{int(upper) if math.isfinite(upper) else 'plus'}"

    @staticmethod
    def _increment_group_fold(
        *,
        anomaly_type: AnomalyType,
        count: int,
        eligible: int,
        n: dict[AnomalyType, int],
        sum_c: dict[AnomalyType, int],
        sum_e: dict[AnomalyType, int],
        sum_c2_over_e: dict[AnomalyType, float],
        sum_c2: dict[AnomalyType, int],
        e_hist: dict[AnomalyType, dict[int, int]],
        bucket_stats: dict[tuple[AnomalyType, str], list[float]],
    ) -> None:
        if eligible <= 0:
            return
        n[anomaly_type] = n.get(anomaly_type, 0) + 1
        sum_c[anomaly_type] = sum_c.get(anomaly_type, 0) + count
        sum_e[anomaly_type] = sum_e.get(anomaly_type, 0) + eligible
        sum_c2_over_e[anomaly_type] = (
            sum_c2_over_e.get(anomaly_type, 0.0) + count * count / eligible
        )
        sum_c2[anomaly_type] = sum_c2.get(anomaly_type, 0) + count * count
        histogram = e_hist.setdefault(anomaly_type, {})
        histogram[eligible] = histogram.get(eligible, 0) + 1
        bucket = bucket_stats.setdefault(
            (anomaly_type, MetricsCollector._occupancy_bucket(eligible)),
            [0.0, 0.0, 0.0],
        )
        bucket[0] += 1
        bucket[1] += count
        bucket[2] += count * count

    @staticmethod
    def _background_group_sort_key(
        key: tuple[int, AnomalyType],
    ) -> tuple[int, str]:
        """Return a process-stable order for background groups."""
        zone_id, anomaly_type = key
        return zone_id, anomaly_type.value

    def _live_fold_state(self) -> _BackgroundFoldState:
        """View the live background accumulators as one fold state.

        Every dict is shared by reference, so folding through the view mutates
        the collector's own counters; only the two rebound fields (the open bin
        and its group table) need writing back afterwards.
        """
        return _BackgroundFoldState(
            open_time_bin=self._background_open_time_bin,
            open_groups=self._background_open_groups,
            emission_n=self._background_emission_n,
            emission_sum_c=self._background_emission_sum_c,
            emission_sum_e=self._background_emission_sum_e,
            emission_sum_c2_over_e=self._background_emission_sum_c2_over_e,
            emission_sum_c2=self._background_emission_sum_c2,
            emission_e_hist=self._background_emission_e_hist,
            emission_bucket_stats=self._background_emission_bucket_stats,
            emission_observed=self._background_emission_observed,
            window_history=self._background_window_history,
            window_n=self._background_window_n,
            window_sum_c=self._background_window_sum_c_fold,
            window_sum_e=self._background_window_sum_e_fold,
            window_sum_c2_over_e=self._background_window_sum_c2_over_e,
            window_sum_c2=self._background_window_sum_c2,
            window_e_hist=self._background_window_e_hist,
            window_bucket_stats=self._background_window_bucket_stats,
            window_observed=self._background_window_observed,
        )

    def _finalize_background_bin(self) -> None:
        """Fold one emission bin and one aggregation-window observation."""
        state = self._live_fold_state()
        self._finalize_background_state(state)
        self._background_open_groups = state.open_groups
        self._background_open_time_bin = state.open_time_bin

    def _finalize_background_state(self, state: _BackgroundFoldState) -> None:
        """Fold one bin of `state`, closing it out."""
        if state.open_time_bin is None:
            return
        current = state.open_groups
        keys = sorted(
            set(state.window_history) | set(current),
            key=self._background_group_sort_key,
        )
        for zone_id, anomaly_type in keys:
            count, eligible = current.get((zone_id, anomaly_type), [0, 0])
            self._increment_group_fold(
                anomaly_type=anomaly_type,
                count=count,
                eligible=eligible,
                n=state.emission_n,
                sum_c=state.emission_sum_c,
                sum_e=state.emission_sum_e,
                sum_c2_over_e=state.emission_sum_c2_over_e,
                sum_c2=state.emission_sum_c2,
                e_hist=state.emission_e_hist,
                bucket_stats=state.emission_bucket_stats,
            )
            if self.aggregation_threshold is not None and count >= self.aggregation_threshold:
                state.emission_observed[anomaly_type] = (
                    state.emission_observed.get(anomaly_type, 0) + 1
                )
            history = state.window_history.setdefault(
                (zone_id, anomaly_type),
                deque(maxlen=self.aggregation_window_bins),
            )
            history.append((count, eligible))
            window_count = sum(item[0] for item in history)
            window_eligible = sum(item[1] for item in history)
            self._increment_group_fold(
                anomaly_type=anomaly_type,
                count=window_count,
                eligible=window_eligible,
                n=state.window_n,
                sum_c=state.window_sum_c,
                sum_e=state.window_sum_e,
                sum_c2_over_e=state.window_sum_c2_over_e,
                sum_c2=state.window_sum_c2,
                e_hist=state.window_e_hist,
                bucket_stats=state.window_bucket_stats,
            )
            if (
                self.aggregation_threshold is not None
                and window_count >= self.aggregation_threshold
            ):
                state.window_observed[anomaly_type] = state.window_observed.get(anomaly_type, 0) + 1
                history.clear()
                history.append((count, eligible))
            elif not any(history):
                state.window_history.pop((zone_id, anomaly_type), None)
        state.open_groups = {}
        state.open_time_bin = None

    @staticmethod
    def _poisson_tail(lam: float, threshold: int) -> float:
        if lam <= 0:
            return 0.0
        probability = math.exp(-lam)
        cumulative = probability
        for count in range(1, threshold):
            probability *= lam / count
            cumulative += probability
        return max(0.0, min(1.0, 1.0 - cumulative))

    def _background_agent_summary(self) -> dict[str, object]:
        counts = self._background_agent_tokens
        if not counts:
            return {
                "background_agent_count": 0,
                "background_agent_mean_tokens": None,
                "background_agent_sd_tokens": None,
                "background_agent_variance_to_mean": None,
                "background_agent_top_decile_token_share": None,
                "background_agent_zero_fraction": None,
                "background_agent_max_tokens": None,
            }
        mean = sum(counts) / len(counts)
        sum_sq = sum(count * count for count in counts)
        variance = sum_sq / len(counts) - mean * mean
        top_count = max(1, math.ceil(len(counts) / 10))
        top_tokens = sum(sorted(counts, reverse=True)[:top_count])
        return {
            "background_agent_count": len(counts),
            "background_agent_mean_tokens": mean,
            "background_agent_sd_tokens": math.sqrt(max(0.0, variance)),
            "background_agent_variance_to_mean": variance / mean if mean else None,
            "background_agent_top_decile_token_share": (
                top_tokens / sum(counts) if sum(counts) else None
            ),
            "background_agent_zero_fraction": sum(count == 0 for count in counts) / len(counts),
            "background_agent_max_tokens": max(counts),
        }

    def _background_assessment(self) -> dict[str, object]:
        """Finalize emission- and aggregation-window background assessments."""
        self._finalize_background_bin()
        self._finalize_background_state(self._background_settled)
        if self._background_assessment_finalized is not None:
            return self._background_assessment_finalized

        def rates_for(
            tokens: dict[AnomalyType, int], eligible: dict[AnomalyType, int]
        ) -> dict[AnomalyType, float]:
            return {
                anomaly_type: (
                    tokens.get(anomaly_type, 0) / eligible.get(anomaly_type, 1)
                    if eligible.get(anomaly_type, 0) > 0
                    else 0.0
                )
                for anomaly_type in AnomalyType
            }

        rates = rates_for(self._background_type_tokens, self._background_type_eligible)
        settled_rates = rates_for(
            self._background_settled_type_tokens,
            self._background_settled_type_eligible,
        )

        def finalize_fold(
            rates: dict[AnomalyType, float],
            n_by_type: dict[AnomalyType, int],
            sum_c_by_type: dict[AnomalyType, int],
            sum_e_by_type: dict[AnomalyType, int],
            sum_c2_over_e_by_type: dict[AnomalyType, float],
            e_hist_by_type: dict[AnomalyType, dict[int, int]],
            bucket_stats_by_type: dict[tuple[AnomalyType, str], list[float]],
            observed_by_type: dict[AnomalyType, int],
        ) -> dict[str, object]:
            valid_types = [anomaly_type for anomaly_type in AnomalyType if rates[anomaly_type] > 0]
            n_groups = sum(n_by_type.get(anomaly_type, 0) for anomaly_type in valid_types)
            sum_counts = sum(sum_c_by_type.get(anomaly_type, 0) for anomaly_type in valid_types)
            sum_lambda = sum(
                rates[anomaly_type] * sum_e_by_type.get(anomaly_type, 0)
                for anomaly_type in valid_types
            )
            numerator = sum(
                sum_c2_over_e_by_type.get(anomaly_type, 0.0) / rates[anomaly_type]
                - 2 * sum_c_by_type.get(anomaly_type, 0)
                + rates[anomaly_type] * sum_e_by_type.get(anomaly_type, 0)
                for anomaly_type in valid_types
            )
            bucket_results: dict[str, dict[str, float | int | None]] = {}
            for (anomaly_type, label), (
                bucket_n,
                bucket_sum,
                bucket_sum_sq,
            ) in bucket_stats_by_type.items():
                mean = bucket_sum / bucket_n
                bucket_results[f"{anomaly_type.value}:{label}"] = {
                    "n_groups": int(bucket_n),
                    "sum_counts": int(bucket_sum),
                    "variance_to_mean": (
                        (bucket_sum_sq / bucket_n - mean * mean) / mean if mean else None
                    ),
                }
            threshold = self.aggregation_threshold
            observed_fraction = (
                sum(observed_by_type.values()) / n_groups
                if threshold is not None and n_groups
                else None
            )
            predicted_fraction = (
                sum(
                    self._poisson_tail(rates[anomaly_type] * eligible, threshold) * count
                    for anomaly_type, histogram in e_hist_by_type.items()
                    for eligible, count in histogram.items()
                )
                / n_groups
                if threshold is not None and n_groups
                else None
            )
            excluded = sum(
                n_by_type.get(anomaly_type, 0)
                for anomaly_type in AnomalyType
                if rates[anomaly_type] <= 0
            )
            return {
                "group_count": n_groups,
                "group_count_excluded_lambda_zero": excluded,
                "group_sum_counts": sum_counts,
                "group_mean_lambda": sum_lambda / n_groups if n_groups else None,
                "pearson_dispersion": numerator / n_groups if n_groups else None,
                "occupancy_buckets": bucket_results,
                "groups_observed_at_threshold_fraction": observed_fraction,
                "groups_poisson_tail_fraction": predicted_fraction,
            }

        emission = finalize_fold(
            rates,
            self._background_emission_n,
            self._background_emission_sum_c,
            self._background_emission_sum_e,
            self._background_emission_sum_c2_over_e,
            self._background_emission_e_hist,
            self._background_emission_bucket_stats,
            self._background_emission_observed,
        )
        window = finalize_fold(
            rates,
            self._background_window_n,
            self._background_window_sum_c_fold,
            self._background_window_sum_e_fold,
            self._background_window_sum_c2_over_e,
            self._background_window_e_hist,
            self._background_window_bucket_stats,
            self._background_window_observed,
        )
        settled_emission = finalize_fold(
            settled_rates,
            self._background_settled.emission_n,
            self._background_settled.emission_sum_c,
            self._background_settled.emission_sum_e,
            self._background_settled.emission_sum_c2_over_e,
            self._background_settled.emission_e_hist,
            self._background_settled.emission_bucket_stats,
            self._background_settled.emission_observed,
        )
        settled_window = finalize_fold(
            settled_rates,
            self._background_settled.window_n,
            self._background_settled.window_sum_c,
            self._background_settled.window_sum_e,
            self._background_settled.window_sum_c2_over_e,
            self._background_settled.window_e_hist,
            self._background_settled.window_bucket_stats,
            self._background_settled.window_observed,
        )

        def population_vmr(n: int, total: int, total_sq: int) -> float | None:
            if n == 0 or total == 0:
                return None
            mean = total / n
            return (total_sq / n - mean * mean) / mean

        result: dict[str, object] = {
            "background_rate_by_anomaly_type": {
                anomaly_type.value: rates[anomaly_type] for anomaly_type in AnomalyType
            },
            "background_rate": (
                self._background_total_tokens / self._background_total_eligible
                if self._background_total_eligible > 0
                else None
            ),
            "background_emission_group_count": emission["group_count"],
            "background_emission_group_count_excluded_lambda_zero": emission[
                "group_count_excluded_lambda_zero"
            ],
            "background_emission_group_sum_counts": emission["group_sum_counts"],
            "background_emission_group_mean_lambda": emission["group_mean_lambda"],
            "background_emission_pearson_dispersion": emission["pearson_dispersion"],
            "background_emission_occupancy_buckets": emission["occupancy_buckets"],
            "background_emission_observed_at_threshold_fraction": emission[
                "groups_observed_at_threshold_fraction"
            ],
            "background_emission_poisson_tail_fraction": emission["groups_poisson_tail_fraction"],
            "background_window_group_count": window["group_count"],
            "background_window_group_count_excluded_lambda_zero": window[
                "group_count_excluded_lambda_zero"
            ],
            "background_window_group_sum_counts": window["group_sum_counts"],
            "background_window_group_mean_lambda": window["group_mean_lambda"],
            "background_window_pearson_dispersion": window["pearson_dispersion"],
            "background_window_occupancy_buckets": window["occupancy_buckets"],
            "background_window_observed_at_threshold_fraction": window[
                "groups_observed_at_threshold_fraction"
            ],
            "background_window_poisson_tail_fraction": window["groups_poisson_tail_fraction"],
            # Backward-compatible aliases for the original emission-level names.
            "background_group_count": emission["group_count"],
            "background_group_count_excluded_lambda_zero": emission[
                "group_count_excluded_lambda_zero"
            ],
            "background_group_sum_counts": emission["group_sum_counts"],
            "background_group_mean_lambda": emission["group_mean_lambda"],
            "background_pearson_dispersion": emission["pearson_dispersion"],
            "background_occupancy_buckets": emission["occupancy_buckets"],
            "background_groups_observed_at_threshold_fraction": emission[
                "groups_observed_at_threshold_fraction"
            ],
            "background_groups_poisson_tail_fraction": emission["groups_poisson_tail_fraction"],
            "background_window_length_bins": self.aggregation_window_bins,
            "background_settled_rate_by_anomaly_type": {
                anomaly_type.value: settled_rates[anomaly_type] for anomaly_type in AnomalyType
            },
            "background_settled_rate": (
                self._background_settled_tokens / self._background_settled_eligible
                if self._background_settled_eligible > 0
                else None
            ),
            "background_settled_emission_group_count": settled_emission["group_count"],
            "background_settled_emission_group_count_excluded_lambda_zero": (
                settled_emission["group_count_excluded_lambda_zero"]
            ),
            "background_settled_emission_group_sum_counts": settled_emission["group_sum_counts"],
            "background_settled_emission_group_mean_lambda": settled_emission["group_mean_lambda"],
            "background_settled_emission_pearson_dispersion": settled_emission[
                "pearson_dispersion"
            ],
            "background_settled_emission_occupancy_buckets": settled_emission["occupancy_buckets"],
            "background_settled_emission_observed_at_threshold_fraction": (
                settled_emission["groups_observed_at_threshold_fraction"]
            ),
            "background_settled_emission_poisson_tail_fraction": settled_emission[
                "groups_poisson_tail_fraction"
            ],
            "background_settled_window_group_count": settled_window["group_count"],
            "background_settled_window_group_count_excluded_lambda_zero": (
                settled_window["group_count_excluded_lambda_zero"]
            ),
            "background_settled_window_group_sum_counts": settled_window["group_sum_counts"],
            "background_settled_window_group_mean_lambda": settled_window["group_mean_lambda"],
            "background_settled_window_pearson_dispersion": settled_window["pearson_dispersion"],
            "background_settled_window_occupancy_buckets": settled_window["occupancy_buckets"],
            "background_settled_window_observed_at_threshold_fraction": (
                settled_window["groups_observed_at_threshold_fraction"]
            ),
            "background_settled_window_poisson_tail_fraction": settled_window[
                "groups_poisson_tail_fraction"
            ],
            "background_settled_window_length_bins": self.aggregation_window_bins,
            "background_population_variance_to_mean": population_vmr(
                self._background_population_n,
                self._background_population_sum,
                self._background_population_sum_sq,
            ),
            "background_settled_population_variance_to_mean": population_vmr(
                self._background_settled_population_n,
                self._background_settled_population_sum,
                self._background_settled_population_sum_sq,
            ),
            **self._background_agent_summary(),
        }
        self._background_assessment_finalized = result
        return result

    cause_attributed_detections: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"disease": {}, "toxin": {}}
    )
    _step_cause_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"disease": {}, "toxin": {}}
    )

    @staticmethod
    def _cause_bucket(hazard_type: str, cause: PerturbationCause) -> str:
        return "hazard" if cause.value == hazard_type else cause.value

    @staticmethod
    def _cause_buckets(hazard_type: str) -> tuple[str, ...]:
        return ("hazard", "none") + tuple(
            cause.value for cause in PerturbationCause if cause.value != hazard_type
        )

    def _record_cause_counts(self, event: DetectionEvent) -> None:
        causes = event.causes
        if not causes:
            buckets = {"none"}
        else:
            buckets = {self._cause_bucket(event.hazard_type, cause) for cause in causes}
        for bucket in sorted(buckets):
            self.cause_attributed_detections[event.hazard_type][bucket] = (
                self.cause_attributed_detections[event.hazard_type].get(bucket, 0) + 1
            )
            self._step_cause_counts[event.hazard_type][bucket] = (
                self._step_cause_counts[event.hazard_type].get(bucket, 0) + 1
            )

    def record_baseline_warmup_config(self, steps: int) -> None:
        """Store configured baseline warm-up length for summary output."""
        self.baseline_warmup_steps = steps

    def record_baseline_maturation(
        self,
        *,
        minimum_history_days: int,
        maximum_history_days: int,
        cadence_steps: int,
        fleet_start_count: int,
        history_days: np.ndarray,
        sample_counts: np.ndarray,
        circadian_bins: np.ndarray,
        monthly_bins: np.ndarray,
    ) -> None:
        """Store device-local maturation summaries for evaluation output."""
        self.baseline_maturation_minimum_history_days = minimum_history_days
        self.baseline_maturation_maximum_history_days = maximum_history_days
        self.baseline_maturation_cadence_steps = cadence_steps
        self.baseline_maturation_device_count = fleet_start_count
        self.baseline_maturation_history_days = history_days.astype(int).tolist()
        self.baseline_maturation_sample_counts = sample_counts.astype(int).tolist()
        self.baseline_maturation_circadian_bins = circadian_bins.astype(int).tolist()
        self.baseline_maturation_monthly_bins = monthly_bins.astype(int).tolist()

    def record_world_settling_config(self, steps: int) -> None:
        """Store the configured world-settling exclusion length."""
        self.world_settling_steps = steps

    def record_fleet_cold_start(self, cold_start: bool) -> None:
        """Record whether cold-baseline behavior reached the protocol."""
        self.fleet_cold_start = self.fleet_cold_start or cold_start

    def record_device_re_adoption(self, legacy_warmup_reset: bool) -> None:
        """Record a device return to ACTIVE and any legacy reset applied."""
        self.device_re_adoption_count += 1
        if legacy_warmup_reset:
            self.legacy_device_adoption_warmup_reset_count += 1

    def record_device_adoption(self, step: int, zone_id: int) -> None:
        """Record a first-time adoption event and its current zone."""
        self.adoption_events.append({"step": step, "zone_id": zone_id})

    def record_population_config(self, n_agents: int) -> None:
        """Store population size for per-agent operational metrics."""
        self.n_agents = n_agents

    def record_anomaly_threshold_config(self, threshold: float) -> None:
        """Store the configured anomaly threshold for reproducibility."""
        self.anomaly_threshold = threshold

    def record_aggregation_threshold_config(self, threshold: int) -> None:
        """Store the aggregation threshold for background tail comparisons."""
        self.aggregation_threshold = threshold

    def record_aggregation_window_config(self, bins: int) -> None:
        """Store the rolling aggregation window length in timestamp bins."""
        self.aggregation_window_bins = bins

    def record_detector_config(
        self,
        mode: str,
        reference_value: float,
        threshold: float,
        clear_steps: int,
        clear_fraction: float,
        residual_ewma_alpha: float,
    ) -> None:
        """Store detector settings for reproducible summary output."""
        self.detector_mode = mode
        self.sequential_reference_value = reference_value
        self.sequential_threshold = threshold
        self.sequential_clear_steps = clear_steps
        self.sequential_clear_fraction = clear_fraction
        self.sequential_residual_ewma_alpha = residual_ewma_alpha

    def warmup_step_count(self) -> int:
        """Count simulation steps that occurred during global baseline warm-up."""
        return sum(1 for record in self.step_records if record.get("baseline_warmup_active"))

    def _settlement_marker_summary(self) -> dict[str, object]:
        """Return world-settling, fleet-cold-start, and onboarding markers."""
        run_steps = len(self.step_records)
        settling_steps = max(0, self.world_settling_steps)
        steps_before = min(run_steps, settling_steps)
        steps_after = max(0, run_steps - settling_steps)
        wearable_denominator = self._wearable_steps
        post_settling_denominator = self._post_world_settling_wearable_steps
        return {
            "world_settling_steps": settling_steps,
            "world_settling_complete": run_steps > settling_steps,
            "world_settling_status": ("settled" if run_steps > settling_steps else "not_settled"),
            "steps_before_world_settling": steps_before,
            "steps_after_world_settling": steps_after,
            "world_settling_fraction_of_run": (steps_before / run_steps if run_steps else None),
            "fleet_cold_start": self.fleet_cold_start,
            "fleet_cold_baseline_wearable_step_fraction": (
                self._cold_baseline_wearable_steps / wearable_denominator
                if wearable_denominator
                else None
            ),
            "post_world_settling_cold_baseline_wearable_step_fraction": (
                self._post_world_settling_cold_baseline_wearable_steps / post_settling_denominator
                if post_settling_denominator
                else None
            ),
            "device_re_adoption_count": self.device_re_adoption_count,
            "legacy_device_adoption_warmup_reset_count": (
                self.legacy_device_adoption_warmup_reset_count
            ),
            "adoption_events": list(self.adoption_events),
            "peak_onboarding_cold_wearables_in_zone": (self.peak_onboarding_cold_wearables_in_zone),
            "peak_onboarding_wearables_in_zone": (self.peak_onboarding_wearables_in_zone),
        }

    # Episode state for FN/TN counting (one FN or TN per episode, not per step)
    _disease_episode: _HazardEpisodeState = field(default_factory=_HazardEpisodeState)
    _toxin_episode: _HazardEpisodeState = field(default_factory=_HazardEpisodeState)

    def _episode_state(self, hazard_type: str) -> _HazardEpisodeState:
        if hazard_type == "disease":
            return self._disease_episode
        if hazard_type == "toxin":
            return self._toxin_episode
        raise ValueError(f"Unknown hazard type: {hazard_type}")

    def record_disease_onset(self, step: int, instance_id: str | None = None) -> None:
        """Mark the step when first infectious case appears."""
        if self.disease_onset_step is None:
            self.disease_onset_step = step
        if instance_id and instance_id not in self.disease_onset_steps:
            self.disease_onset_steps[instance_id] = step

    def record_toxin_onset(self, step: int, instance_id: str | None = None) -> None:
        """Mark the step when a plume begins."""
        if self.toxin_onset_step is None:
            self.toxin_onset_step = step
        if instance_id and instance_id not in self.toxin_onset_steps:
            self.toxin_onset_steps[instance_id] = step

    def _record_true_positive(self, event: DetectionEvent) -> None:
        if event.hazard_type == "disease":
            self.true_positives_disease += 1
            if self.disease_detection_step is None:
                self.disease_detection_step = event.step
        elif event.hazard_type == "toxin":
            self.true_positives_toxin += 1
            if self.toxin_detection_step is None:
                self.toxin_detection_step = event.step
        if event.hazard_instance_id:
            key = f"{event.hazard_type}:{event.hazard_instance_id}"
            self.instance_true_positives[key] = self.instance_true_positives.get(key, 0) + 1

    def _record_false_positive(self, event: DetectionEvent) -> None:
        if event.hazard_type == "disease":
            self.false_positives_disease += 1
        elif event.hazard_type == "toxin":
            self.false_positives_toxin += 1

    def record_detection(self, event: DetectionEvent) -> None:
        """Record a system detection event and update confusion matrix."""
        self.detection_events.append(event)
        if (
            event.true_positive
            and event.hazard_type == "toxin"
            and event.dosed_agents is not None
            and event.dosed_agents < 2
        ):
            self.toxin_true_positives_fewer_than_two_dosed_agents += 1
        self._record_cause_counts(event)
        self._record_warrant(event)
        if event.benign_instance_id is not None:
            self.benign_overlap_detections += 1
        if event.benign_attributed:
            self.benign_attributed_detections += 1
            if event.true_positive:
                self.benign_coincident_true_positives += 1
            else:
                self.benign_misattributed_detections += 1
                if event.benign_cause in BENIGN_CAUSES:
                    cause = event.benign_cause
                    self.benign_misattributions_by_cause[cause.value] = (
                        self.benign_misattributions_by_cause.get(cause.value, 0) + 1
                    )
        if event.true_positive:
            self._record_true_positive(event)
            if event.attributed is True:
                if event.hazard_type == "disease":
                    self.attributed_true_positives_disease += 1
                    if self.attributed_disease_detection_step is None:
                        self.attributed_disease_detection_step = event.step
                elif event.hazard_type == "toxin":
                    self.attributed_true_positives_toxin += 1
                    if self.attributed_toxin_detection_step is None:
                        self.attributed_toxin_detection_step = event.step
            elif event.attributed is False:
                if event.hazard_type == "disease":
                    self.coincidental_true_positives_disease += 1
                elif event.hazard_type == "toxin":
                    self.coincidental_true_positives_toxin += 1
        else:
            self._record_false_positive(event)

    def _record_warrant(self, event: DetectionEvent) -> None:
        warrant = self._warrant_class(event)
        if warrant is WarrantClass.TARGET:
            self.target_detections += 1
            self.warranted_detections += 1
        elif warrant is WarrantClass.ACTIONABLE_NON_TARGET:
            self.actionable_non_target_detections += 1
            self.warranted_detections += 1
        elif warrant is WarrantClass.EXPLAINED:
            self.explained_detections += 1
        elif warrant is WarrantClass.ARTIFACT:
            self.artifact_detections += 1
        else:
            self.unexplained_detections += 1

    @staticmethod
    def _warrant_class(event: DetectionEvent) -> WarrantClass:
        if event.true_positive:
            return WarrantClass.TARGET
        benign_causes = set(event.benign_causes)
        if event.benign_cause is not None:
            benign_causes.add(event.benign_cause)
        actionable = {
            PerturbationCause.HEAT_WAVE,
            PerturbationCause.BACKGROUND_ILI,
            PerturbationCause.IRRITANT_EXPOSURE,
        }
        if benign_causes & actionable:
            return WarrantClass.ACTIONABLE_NON_TARGET
        if benign_causes:
            return WarrantClass.EXPLAINED
        if PerturbationCause.SENSOR_ARTIFACT in event.causes:
            return WarrantClass.ARTIFACT
        return WarrantClass.UNEXPLAINED

    def record_affected_agent_token(
        self, hazard_type: str, anomaly_type: AnomalyType, group_size: int
    ) -> None:
        """Record provenance-only fragmentation statistics for affected tokens."""
        by_type = self.affected_agent_token_counts[hazard_type]
        key = anomaly_type.value
        by_type[key] = by_type.get(key, 0) + 1
        self.largest_affected_agent_group[hazard_type] = max(
            self.largest_affected_agent_group[hazard_type], group_size
        )

    def record_sybil_false_alert(self, count: int = 1) -> None:
        """Record a false alert triggered by Sybil token injection."""
        self.sybil_false_alerts += count

    def sync_attack_metrics(self, orchestrator) -> None:
        """Sync attack orchestrator counters into summary metrics."""
        self.deanon_attempts = orchestrator.deanon_attempts
        self.deanon_successes = orchestrator.deanon_successes
        self.eclipse_tokens_dropped = orchestrator.eclipse.dropped_count
        self.correlation_evaluations = orchestrator.correlation_evaluations
        self.correlation_successes = orchestrator.correlation_successes
        self.replay_tokens_injected = orchestrator.replay.replay_inject_count
        self.replay_false_alerts = orchestrator.replay_false_alerts

    def record_missed_detection(self, hazard_type: str) -> None:
        """Record one false negative for a completed undetected hazard episode."""
        if hazard_type == "disease":
            self.false_negatives_disease += 1
        elif hazard_type == "toxin":
            self.false_negatives_toxin += 1

    def record_no_hazard_no_detection(self, hazard_type: str) -> None:
        """Record one true negative for a completed no-hazard episode without false alarms."""
        if hazard_type == "disease":
            self.true_negatives_disease += 1
        elif hazard_type == "toxin":
            self.true_negatives_toxin += 1

    def update_hazard_episode(
        self,
        hazard_type: str,
        is_active: bool,
        true_positive_this_step: bool,
        false_positive_this_step: bool,
    ) -> None:
        """Update episode state and close episodes on hazard transitions.

        Each hazard-active episode contributes at most one FN if it ends without
        a true-positive detection. Each no-hazard episode contributes at most one
        TN if it ends without a false-positive detection.
        """
        state = self._episode_state(hazard_type)
        was_active = state.hazard_active

        if was_active and not is_active:
            if not state.detected_in_hazard_episode:
                self.record_missed_detection(hazard_type)
            state.false_positive_in_no_hazard_episode = False

        if not was_active and is_active:
            if not state.false_positive_in_no_hazard_episode:
                self.record_no_hazard_no_detection(hazard_type)
            state.detected_in_hazard_episode = False

        if is_active and true_positive_this_step:
            state.detected_in_hazard_episode = True
        if not is_active and false_positive_this_step:
            state.false_positive_in_no_hazard_episode = True

        state.hazard_active = is_active

    def finalize_hazard_episodes(self) -> None:
        """Close open episodes at simulation end."""
        for hazard_type in ("disease", "toxin"):
            state = self._episode_state(hazard_type)
            if state.hazard_active:
                if not state.detected_in_hazard_episode:
                    self.record_missed_detection(hazard_type)
            elif not state.false_positive_in_no_hazard_episode:
                self.record_no_hazard_no_detection(hazard_type)

    def record_dilation(
        self,
        *,
        dilated_cell_count: int,
        resident_population: int,
        estimated_respondent_population: int,
        true_respondent_population: int,
        k_min: int,
        step: int = 0,
        responding_devices: int = 0,
        release_suppressed: bool = False,
        response_epsilon_burned: float = 0.0,
    ) -> None:
        """Record evaluation-only population measurements for one broadcast."""
        self.dilation_records.append(
            {
                "dilated_cell_count": dilated_cell_count,
                "resident_population": resident_population,
                "estimated_respondent_population": estimated_respondent_population,
                "true_respondent_population": true_respondent_population,
                "true_respondents_meet_k": true_respondent_population >= k_min,
                "step": step,
                "responding_devices": responding_devices,
                "release_suppressed": release_suppressed,
                "response_epsilon_burned": response_epsilon_burned,
            }
        )
        if release_suppressed:
            self.dilation_release_suppressed_records.append(
                {
                    "step": step,
                    "responding_devices": responding_devices,
                    "k_min": k_min,
                    "response_epsilon_burned": response_epsilon_burned,
                }
            )

    def record_dilation_suppressed(
        self,
        *,
        estimated_respondent_population: int,
        k_min: int,
        step: int = 0,
    ) -> None:
        """Record a trigger suppressed because estimated k-anonymity was infeasible."""
        self.dilation_suppressed_records.append(
            {
                "estimated_respondent_population": estimated_respondent_population,
                "k_min": k_min,
                "step": step,
            }
        )

    def _dilation_metrics(self) -> dict[str, float | int | None]:
        """Summarize evaluation-only k-anonymity dilation measurements."""
        if not self.dilation_records and not self.dilation_suppressed_records:
            return {
                "dilation_broadcasts": 0,
                "dilation_suppressed_for_insufficient_anonymity": 0,
                "dilation_suppression_rate": None,
                "suppressed_estimated_respondent_population_mean": None,
                "dilation_release_suppressed_for_insufficient_anonymity": 0,
                "dilation_release_suppression_rate": None,
                "dilation_release_suppressed_epsilon": 0.0,
                "dilation_release_suppressed_epsilon_share": None,
                "estimated_to_true_respondent_ratio_median": None,
                "dilated_cells_mean": None,
                "dilated_cells_p50": None,
                "dilated_cells_p90": None,
                "resident_population_mean": None,
                "resident_population_p50": None,
                "resident_population_p90": None,
                "estimated_respondent_population_mean": None,
                "estimated_respondent_population_p50": None,
                "estimated_respondent_population_p90": None,
                "true_respondent_population_mean": None,
                "true_respondent_population_p50": None,
                "true_respondent_population_p90": None,
                "fraction_true_respondents_meeting_k": None,
            }
        metrics: dict[str, float | int | None] = {
            "dilation_broadcasts": len(self.dilation_records),
            "dilation_suppressed_for_insufficient_anonymity": len(self.dilation_suppressed_records),
            "dilation_release_suppressed_for_insufficient_anonymity": len(
                self.dilation_release_suppressed_records
            ),
            "fraction_true_respondents_meeting_k": (
                float(
                    np.mean([record["true_respondents_meet_k"] for record in self.dilation_records])
                )
                if self.dilation_records
                else None
            ),
        }
        attempts = len(self.dilation_records) + len(self.dilation_suppressed_records)
        metrics["dilation_suppression_rate"] = (
            len(self.dilation_suppressed_records) / attempts if attempts else None
        )
        issued = len(self.dilation_records)
        metrics["dilation_release_suppression_rate"] = (
            len(self.dilation_release_suppressed_records) / issued if issued else None
        )
        suppressed_epsilon = sum(
            float(record["response_epsilon_burned"])
            for record in self.dilation_release_suppressed_records
        )
        total_response_epsilon = sum(
            float(record["response_epsilon_burned"]) for record in self.dilation_records
        )
        metrics["dilation_release_suppressed_epsilon"] = suppressed_epsilon
        metrics["dilation_release_suppressed_epsilon_share"] = (
            suppressed_epsilon / total_response_epsilon if total_response_epsilon else None
        )
        if self.dilation_suppressed_records:
            metrics["suppressed_estimated_respondent_population_mean"] = float(
                np.mean(
                    [
                        record["estimated_respondent_population"]
                        for record in self.dilation_suppressed_records
                    ]
                )
            )
        else:
            metrics["suppressed_estimated_respondent_population_mean"] = None
        metrics.update(self._dilation_population_metrics())
        return metrics

    def _dilation_population_metrics(self) -> dict[str, float | None]:
        """Summarize population and zone-size distributions for issued broadcasts."""
        output_names = (
            "dilated_cells",
            "resident_population",
            "estimated_respondent_population",
            "true_respondent_population",
        )
        if not self.dilation_records:
            empty_metrics: dict[str, float | None] = {
                f"{output_name}_{stat}": None
                for output_name in output_names
                for stat in ("mean", "p50", "p90")
            }
            empty_metrics["estimated_to_true_respondent_ratio_median"] = None
            return empty_metrics

        metrics: dict[str, float | None] = {}
        for field_name, output_name in (
            ("dilated_cell_count", "dilated_cells"),
            ("resident_population", "resident_population"),
            ("estimated_respondent_population", "estimated_respondent_population"),
            ("true_respondent_population", "true_respondent_population"),
        ):
            values = np.asarray(
                [record[field_name] for record in self.dilation_records], dtype=float
            )
            metrics[f"{output_name}_mean"] = float(np.mean(values))
            metrics[f"{output_name}_p50"] = float(np.percentile(values, 50))
            metrics[f"{output_name}_p90"] = float(np.percentile(values, 90))
        ratios = [
            record["estimated_respondent_population"] / record["true_respondent_population"]
            for record in self.dilation_records
            if record["true_respondent_population"] > 0
        ]
        metrics["estimated_to_true_respondent_ratio_median"] = (
            float(np.median(ratios)) if ratios else None
        )
        return metrics

    def record_step(
        self,
        step: int,
        seir_counts: dict[str, int],
        plume_exposed: int,
        anomalies_detected: int,
        tokens_submitted: int,
        broadcasts_issued: int,
        responses_received: int,
        cumulative_epsilon: float,
        sybil_tokens_injected: int = 0,
        replay_tokens_injected: int = 0,
        eclipse_tokens_dropped: int = 0,
        wearables_active: int = 0,
        wearables_offline: int = 0,
        wearables_not_worn: int = 0,
        wearables_powered_off: int = 0,
        wearables_depleted: int = 0,
        mean_battery_level: float = 1.0,
        baseline_warmup_active: bool = False,
        wearables_in_warmup: int = 0,
        background_tokens: int = 0,
        background_eligible_wearables: int = 0,
        background_rate: float | None = None,
        operational_wearables: int = 0,
        cold_baseline_wearables: int = 0,
        not_adopted_wearables: int = 0,
        adopted_wearables: int = 0,
        onboarding_cold_wearables_in_zone: int = 0,
        onboarding_wearables_in_zone: int = 0,
        disambiguation_queries_issued: int = 0,
        disambiguation_asks_suppressed_by_budget: int = 0,
        disambiguation_acks: int = 0,
        disambiguation_ack_release_count: int = 0,
        disambiguation_devices_reached: int = 0,
        disambiguation_yes_answers: int = 0,
        disambiguation_no_answers: int = 0,
        disambiguation_unanswered_expired: int = 0,
        disambiguation_unresolved_hypotheses: int = 0,
        disambiguation_answer_epsilon: float = 0.0,
        disambiguation_ack_epsilon: float = 0.0,
        disambiguation_well_founded_queries: int = 0,
        disambiguation_unfounded_queries: int = 0,
        disambiguation_unscored_queries: int = 0,
        disambiguation_unfounded_ask_epsilon: float = 0.0,
        disambiguation_unscored_ask_epsilon: float = 0.0,
        disambiguation_max_ask_epsilon_delta: float = 0.0,
        disambiguation_well_founded_by_hypothesis: dict[str, int] | None = None,
        disambiguation_unfounded_by_hypothesis: dict[str, int] | None = None,
        disambiguation_unscored_by_hypothesis: dict[str, int] | None = None,
        confounder_contributions: dict[str, int] | None = None,
        confounder_agents_affected: dict[str, set[int]] | None = None,
        heat_wave_active: bool = False,
        heat_wave_instance_id: str | None = None,
        heat_wave_zone_ids: tuple[int, ...] = (),
        heat_wave_start_step: int | None = None,
        heat_wave_end_step: int | None = None,
        occupied_zone_ids: set[int] | None = None,
        alarming_zone_ids: set[int] | None = None,
    ) -> None:
        """Record per-step metrics for CSV output."""
        record = {
            "step": step,
            "time_hours": step * 5 / 60,
            "susceptible": seir_counts.get("S", 0),
            "exposed": seir_counts.get("E", 0),
            "infectious": seir_counts.get("I", 0),
            "recovered": seir_counts.get("R", 0),
            "plume_exposed": plume_exposed,
            "anomalies_detected": anomalies_detected,
            "tokens_submitted": tokens_submitted,
            "broadcasts_issued": broadcasts_issued,
            "responses_received": responses_received,
            "cumulative_epsilon": cumulative_epsilon,
            "sybil_tokens_injected": sybil_tokens_injected,
            "replay_tokens_injected": replay_tokens_injected,
            "eclipse_tokens_dropped": eclipse_tokens_dropped,
            "wearables_active": wearables_active,
            "wearables_offline": wearables_offline,
            "wearables_not_worn": wearables_not_worn,
            "wearables_powered_off": wearables_powered_off,
            "wearables_depleted": wearables_depleted,
            "not_adopted_wearables": not_adopted_wearables,
            "adopted_wearables": adopted_wearables,
            "onboarding_cold_wearables_in_zone": onboarding_cold_wearables_in_zone,
            "onboarding_wearables_in_zone": onboarding_wearables_in_zone,
            "disambiguation_queries_issued": disambiguation_queries_issued,
            "disambiguation_asks_suppressed_by_budget": (disambiguation_asks_suppressed_by_budget),
            "disambiguation_acks": disambiguation_acks,
            "disambiguation_ack_release_count": disambiguation_ack_release_count,
            "disambiguation_devices_reached": disambiguation_devices_reached,
            "disambiguation_yes_answers": disambiguation_yes_answers,
            "disambiguation_no_answers": disambiguation_no_answers,
            "disambiguation_unanswered_expired": disambiguation_unanswered_expired,
            "disambiguation_unresolved_hypotheses": (disambiguation_unresolved_hypotheses),
            "disambiguation_answer_epsilon": disambiguation_answer_epsilon,
            "disambiguation_ack_epsilon": disambiguation_ack_epsilon,
            "disambiguation_well_founded_queries": disambiguation_well_founded_queries,
            "disambiguation_unfounded_queries": disambiguation_unfounded_queries,
            "disambiguation_unscored_queries": disambiguation_unscored_queries,
            "disambiguation_unfounded_ask_epsilon": disambiguation_unfounded_ask_epsilon,
            "disambiguation_unscored_ask_epsilon": disambiguation_unscored_ask_epsilon,
            "disambiguation_max_ask_epsilon_delta": disambiguation_max_ask_epsilon_delta,
            "confounder_contributions": confounder_contributions or {},
            "confounder_agents_affected": {
                cause: len(agents) for cause, agents in (confounder_agents_affected or {}).items()
            },
            "heat_wave_active": heat_wave_active,
            "heat_wave_instance_id": heat_wave_instance_id,
            "mean_battery_level": mean_battery_level,
            "baseline_warmup_active": baseline_warmup_active,
            "wearables_in_warmup": wearables_in_warmup,
            "background_tokens": background_tokens,
            "background_eligible_wearables": background_eligible_wearables,
            "background_rate": background_rate,
            "past_world_settling": step >= self.world_settling_steps,
            "occupied_zones": len(occupied_zone_ids or set()),
            "alarming_zones": len(alarming_zone_ids or set()),
        }
        for hazard_type in ("disease", "toxin"):
            for bucket in self._cause_buckets(hazard_type):
                record[f"{hazard_type}_cause_{bucket}"] = self._step_cause_counts.get(
                    hazard_type, {}
                ).get(bucket, 0)
        self.step_records.append(record)
        self._step_cause_counts = {"disease": {}, "toxin": {}}
        day = step // STEPS_PER_DAY
        self._daily_occupied_zones.setdefault(day, set()).update(occupied_zone_ids or set())
        self._daily_alarming_zones.setdefault(day, set()).update(alarming_zone_ids or set())
        self.epsilon_per_step.append(cumulative_epsilon)
        self.total_queries_issued += broadcasts_issued
        self.total_responses += responses_received
        self.disambiguation_queries_issued += disambiguation_queries_issued
        self.disambiguation_asks_suppressed_by_budget += disambiguation_asks_suppressed_by_budget
        self.disambiguation_acks += disambiguation_acks
        self.disambiguation_ack_release_count += disambiguation_ack_release_count
        self.disambiguation_devices_reached += disambiguation_devices_reached
        self.disambiguation_yes_answers += disambiguation_yes_answers
        self.disambiguation_no_answers += disambiguation_no_answers
        self.disambiguation_unanswered_expired += disambiguation_unanswered_expired
        self.disambiguation_unresolved_hypotheses += disambiguation_unresolved_hypotheses
        self.disambiguation_answer_epsilon = disambiguation_answer_epsilon
        self.disambiguation_ack_epsilon = disambiguation_ack_epsilon
        self.disambiguation_well_founded_queries += disambiguation_well_founded_queries
        self.disambiguation_unfounded_queries += disambiguation_unfounded_queries
        self.disambiguation_unscored_queries += disambiguation_unscored_queries
        self.disambiguation_unfounded_ask_epsilon += disambiguation_unfounded_ask_epsilon
        self.disambiguation_unscored_ask_epsilon += disambiguation_unscored_ask_epsilon
        self.disambiguation_max_ask_epsilon_delta = max(
            self.disambiguation_max_ask_epsilon_delta,
            disambiguation_max_ask_epsilon_delta,
        )
        for hypothesis, count in (disambiguation_well_founded_by_hypothesis or {}).items():
            self.disambiguation_well_founded_by_hypothesis[hypothesis] = (
                self.disambiguation_well_founded_by_hypothesis.get(hypothesis, 0) + count
            )
        for hypothesis, count in (disambiguation_unfounded_by_hypothesis or {}).items():
            self.disambiguation_unfounded_by_hypothesis[hypothesis] = (
                self.disambiguation_unfounded_by_hypothesis.get(hypothesis, 0) + count
            )
        for hypothesis, count in (disambiguation_unscored_by_hypothesis or {}).items():
            self.disambiguation_unscored_by_hypothesis[hypothesis] = (
                self.disambiguation_unscored_by_hypothesis.get(hypothesis, 0) + count
            )
        for cause, count in (confounder_contributions or {}).items():
            self.confounder_contributions_by_cause[cause] = (
                self.confounder_contributions_by_cause.get(cause, 0) + count
            )
        if heat_wave_active:
            self.heat_wave_active_steps += 1
            if heat_wave_instance_id is not None:
                instance = self.heat_wave_instances.setdefault(
                    heat_wave_instance_id,
                    {
                        "instance_id": heat_wave_instance_id,
                        "active_steps": 0,
                        "zone_ids": list(heat_wave_zone_ids),
                        "start_step": heat_wave_start_step,
                        "end_step": heat_wave_end_step,
                    },
                )
                active_steps = instance.get("active_steps", 0)
                if isinstance(active_steps, int):
                    instance["active_steps"] = active_steps + 1
        for cause, agents in (confounder_agents_affected or {}).items():
            self.confounder_agents_affected_by_cause.setdefault(cause, set()).update(agents)
        self._wearable_steps += operational_wearables
        self._cold_baseline_wearable_steps += cold_baseline_wearables
        self.peak_onboarding_cold_wearables_in_zone = max(
            self.peak_onboarding_cold_wearables_in_zone,
            onboarding_cold_wearables_in_zone,
        )
        self.peak_onboarding_wearables_in_zone = max(
            self.peak_onboarding_wearables_in_zone,
            onboarding_wearables_in_zone,
        )
        if step >= self.world_settling_steps:
            self._post_world_settling_wearable_steps += operational_wearables
            self._post_world_settling_cold_baseline_wearable_steps += cold_baseline_wearables

    def time_to_detection_disease(self) -> float | None:
        """Time (in 5-min steps) from disease onset to detection."""
        if self.disease_onset_step is None or self.disease_detection_step is None:
            return None
        latency = self.disease_detection_step - self.disease_onset_step
        return float(latency) if latency >= 0 else None

    def time_to_detection_toxin(self) -> float | None:
        """Time (in 5-min steps) from toxin onset to detection."""
        if self.toxin_onset_step is None or self.toxin_detection_step is None:
            return None
        latency = self.toxin_detection_step - self.toxin_onset_step
        return float(latency) if latency >= 0 else None

    def attributed_time_to_detection_disease(self) -> float | None:
        """Time (in 5-min steps) from disease onset to attributed detection."""
        if self.disease_onset_step is None or self.attributed_disease_detection_step is None:
            return None
        latency = self.attributed_disease_detection_step - self.disease_onset_step
        return float(latency) if latency >= 0 else None

    def attributed_time_to_detection_toxin(self) -> float | None:
        """Time (in 5-min steps) from toxin onset to attributed detection."""
        if self.toxin_onset_step is None or self.attributed_toxin_detection_step is None:
            return None
        latency = self.attributed_toxin_detection_step - self.toxin_onset_step
        return float(latency) if latency >= 0 else None

    def coincidental_fraction(self, hazard_type: str) -> float | None:
        """Return the fraction of hazard true positives that were coincidental."""
        if hazard_type == "disease":
            coincidental = self.coincidental_true_positives_disease
            attributed = self.attributed_true_positives_disease
        elif hazard_type == "toxin":
            coincidental = self.coincidental_true_positives_toxin
            attributed = self.attributed_true_positives_toxin
        else:
            raise ValueError(f"Unknown hazard type: {hazard_type}")
        total = attributed + coincidental
        return coincidental / total if total else None

    def false_positive_rate_disease(self) -> float | None:
        """FPR = FP / (FP + TN) for disease detection.

        TN counts no-hazard episodes without false alarms (at most one per episode).
        """
        denom = self.false_positives_disease + self.true_negatives_disease
        return self.false_positives_disease / denom if denom > 0 else None

    def false_negative_rate_disease(self) -> float | None:
        """FNR = FN / (FN + TP) for disease detection.

        FN counts hazard-active episodes without a true-positive detection
        (at most one per episode). TP counts confirmed true-positive events.
        """
        denom = self.false_negatives_disease + self.true_positives_disease
        return self.false_negatives_disease / denom if denom > 0 else None

    def false_positive_rate_toxin(self) -> float | None:
        """FPR = FP / (FP + TN) for toxin detection.

        TN counts no-hazard episodes without false alarms (at most one per episode).
        """
        denom = self.false_positives_toxin + self.true_negatives_toxin
        return self.false_positives_toxin / denom if denom > 0 else None

    def false_negative_rate_toxin(self) -> float | None:
        """FNR = FN / (FN + TP) for toxin detection.

        FN counts hazard-active episodes without a true-positive detection
        (at most one per episode). TP counts confirmed true-positive events.
        """
        denom = self.false_negatives_toxin + self.true_positives_toxin
        return self.false_negatives_toxin / denom if denom > 0 else None

    def cardiac_detection_count(self) -> int:
        """Count detection events for cardiac anomaly broadcasts."""
        return sum(
            1 for event in self.detection_events if event.anomaly_type == AnomalyType.CARDIAC
        )

    def discrimination_score(self) -> float | None:
        """Measures system's ability to decouple toxin from disease.

        Score in [0, 1] where 1 = perfect discrimination.
        Based on correct classification of anomaly types.
        """
        correct = 0
        total = 0
        for event in self.detection_events:
            total += 1
            disease_match = event.hazard_type == "disease" and event.anomaly_type in (
                AnomalyType.FEBRILE,
                AnomalyType.MULTI_SYSTEM,
                AnomalyType.CARDIAC,
            )
            toxin_match = event.hazard_type == "toxin" and event.anomaly_type in (
                AnomalyType.RESPIRATORY,
                AnomalyType.CARDIAC,
            )
            if disease_match or toxin_match:
                correct += 1
        hazard_detections = {
            hazard_type
            for hazard_type in ("disease", "toxin")
            if any(
                event.hazard_type == hazard_type and event.true_positive
                for event in self.detection_events
            )
        }
        if len(hazard_detections) < 2:
            return None
        return correct / total if total > 0 else None

    def _operational_daily_metrics(self) -> dict[str, dict[str, float | int | None]]:
        """Aggregate operator-facing alert metrics by simulated day."""
        daily: dict[str, dict[str, float | int | None]] = {}
        broadcasts_by_day: dict[int, int] = {}
        for record in self.step_records:
            day = int(record["step"]) // STEPS_PER_DAY
            broadcasts_by_day[day] = broadcasts_by_day.get(day, 0) + int(
                record["broadcasts_issued"]
            )
        for day in sorted(self._daily_occupied_zones):
            broadcasts = broadcasts_by_day.get(day, 0)
            occupied = len(self._daily_occupied_zones[day])
            alarming = len(self._daily_alarming_zones.get(day, set()))
            daily[str(day)] = {
                "broadcasts": broadcasts,
                "occupied_zones": occupied,
                "broadcasts_per_occupied_zone_per_day": (
                    broadcasts / occupied if occupied else 0.0
                ),
                "broadcasts_per_1000_agents_per_day": (
                    broadcasts / self.n_agents * 1000 if self.n_agents else None
                ),
                "fraction_occupied_zones_alarming": (alarming / occupied if occupied else None),
                "alarming_zones": alarming,
            }
            if not occupied:
                daily[str(day)]["broadcasts_per_occupied_zone_per_day"] = None
        return daily

    def _background_daily_metrics(self) -> dict[str, dict[str, float | int | None]]:
        """Return post-warmup background rates by simulation day."""
        daily: dict[str, dict[str, float | int | None]] = {}
        for day in sorted(
            set(self._background_daily_tokens) | set(self._background_daily_eligible)
        ):
            tokens = self._background_daily_tokens.get(day, 0)
            eligible = self._background_daily_eligible.get(day, 0)
            daily[str(day)] = {
                "background_tokens": tokens,
                "eligible_wearable_steps": eligible,
                "background_rate": tokens / eligible if eligible else None,
            }
        return daily

    def configure_privacy_accounting(
        self,
        *,
        response_mechanism: str,
        response_basis: str,
        response_epsilon: float,
        unaffected_positive_probability: float | None,
        aggregate_count_epsilon_per_release: float,
        aggregate_count_evidence_threshold: int,
        aggregate_count_minimum_releasable_count: int,
        geo_epsilon_per_metre: float,
        geo_basis: str,
        ack_basis: str,
    ) -> None:
        """Store declared per-response channel accounting quantities."""
        self.response_mechanism = response_mechanism
        self.response_epsilon_basis = response_basis
        self.response_epsilon_per_response = response_epsilon
        self.unaffected_positive_reply_probability = unaffected_positive_probability
        self.aggregate_count_epsilon_per_release = aggregate_count_epsilon_per_release
        self.aggregate_count_evidence_threshold = aggregate_count_evidence_threshold
        self.aggregate_count_minimum_releasable_count = aggregate_count_minimum_releasable_count
        self.geo_epsilon_per_metre = geo_epsilon_per_metre
        self.geo_epsilon_basis = geo_basis
        self.disambiguation_ack_epsilon_basis = ack_basis

    def record_aggregate_count_release(
        self,
        *,
        query_id: int,
        released_count: int,
        true_count: int,
        population: int,
        epsilon: float,
        composed_epsilon: float,
        evidence_threshold: int,
    ) -> None:
        """Record protocol-visible count and evaluation-only true count."""
        self.aggregate_count_epsilon_per_release = epsilon
        self.aggregate_count_epsilon = composed_epsilon
        self.aggregate_count_composed_epsilon = composed_epsilon
        if released_count == 0:
            self.aggregate_count_no_cluster_releases += 1
            outcome = "no_cluster"
        elif released_count <= evidence_threshold:
            self.aggregate_count_below_floor_releases += 1
            outcome = "cluster_below_floor"
        else:
            self.aggregate_count_evidence_releases += 1
            outcome = "evidence"
        self.aggregate_count_releases.append(
            {
                "query_id": query_id,
                "released_count": released_count,
                "true_count_evaluation_only": true_count,
                "population": population,
                "epsilon": epsilon,
                "composed_epsilon": composed_epsilon,
                "evidence_threshold": evidence_threshold,
                "outcome": outcome,
            }
        )

    def summary(self) -> dict:
        """Generate summary metrics dictionary."""
        ttd_disease = self.time_to_detection_disease()
        ttd_toxin = self.time_to_detection_toxin()
        attributed_ttd_disease = self.attributed_time_to_detection_disease()
        attributed_ttd_toxin = self.attributed_time_to_detection_toxin()
        daily_operational = self._operational_daily_metrics()
        daily_background = self._background_daily_metrics()
        background = self._background_assessment()
        total_true_positives = self.true_positives_disease + self.true_positives_toxin
        issued_precision = (
            total_true_positives / self.total_queries_issued
            if self.total_queries_issued > 0
            else None
        )
        elapsed_days = len(self.step_records) / STEPS_PER_DAY
        epsilon_per_agent_per_day = (
            (self.epsilon_per_step[-1] if self.epsilon_per_step else 0.0)
            / self.n_agents
            / elapsed_days
            if self.n_agents and elapsed_days
            else None
        )
        complete_day_count = len(self.step_records) // STEPS_PER_DAY
        latest_complete_day = str(complete_day_count - 1) if complete_day_count else None
        latest_day = (
            daily_operational.get(latest_complete_day, {})
            if latest_complete_day is not None
            else {}
        )
        cause_counts = {
            hazard_type: {
                bucket: counts.get(bucket, 0) for bucket in self._cause_buckets(hazard_type)
            }
            for hazard_type, counts in self.cause_attributed_detections.items()
        }
        cause_rates: dict[str, dict[str, float | None]] = {}
        for hazard_type, counts in cause_counts.items():
            denominator = sum(event.hazard_type == hazard_type for event in self.detection_events)
            cause_rates[hazard_type] = {
                bucket: (count / denominator if denominator else None)
                for bucket, count in counts.items()
            }
        scorable = {
            hypothesis: self.disambiguation_well_founded_by_hypothesis.get(hypothesis, 0)
            + self.disambiguation_unfounded_by_hypothesis.get(hypothesis, 0)
            for hypothesis in sorted(
                set(self.disambiguation_well_founded_by_hypothesis)
                | set(self.disambiguation_unfounded_by_hypothesis)
            )
        }
        precision_by_hypothesis = {
            hypothesis: self.disambiguation_well_founded_by_hypothesis.get(hypothesis, 0)
            / denominator
            for hypothesis, denominator in scorable.items()
            if denominator
        }
        total_scorable = (
            self.disambiguation_well_founded_queries + self.disambiguation_unfounded_queries
        )
        disambiguation_precision = (
            self.disambiguation_well_founded_queries / total_scorable if total_scorable else None
        )
        return {
            "time_to_detection_disease_steps": ttd_disease,
            "time_to_detection_disease_hours": (
                ttd_disease * 5 / 60 if ttd_disease is not None else None
            ),
            "time_to_detection_toxin_steps": ttd_toxin,
            "time_to_detection_toxin_hours": (
                ttd_toxin * 5 / 60 if ttd_toxin is not None else None
            ),
            "attributed_disease_detections": self.attributed_true_positives_disease,
            "attributed_toxin_detections": self.attributed_true_positives_toxin,
            "coincidental_disease_detections": self.coincidental_true_positives_disease,
            "coincidental_toxin_detections": self.coincidental_true_positives_toxin,
            "attributed_disease_latency_steps": attributed_ttd_disease,
            "attributed_toxin_latency_steps": attributed_ttd_toxin,
            "coincidental_fraction_disease": self.coincidental_fraction("disease"),
            "coincidental_fraction_toxin": self.coincidental_fraction("toxin"),
            "affected_agent_token_counts": self.affected_agent_token_counts,
            "largest_affected_agent_group": self.largest_affected_agent_group,
            "fpr_disease": self.false_positive_rate_disease(),
            "fnr_disease": self.false_negative_rate_disease(),
            "fpr_toxin": self.false_positive_rate_toxin(),
            "fnr_toxin": self.false_negative_rate_toxin(),
            "discrimination_score": self.discrimination_score(),
            "detection_event_counts": {
                "disease": sum(event.hazard_type == "disease" for event in self.detection_events),
                "toxin": sum(event.hazard_type == "toxin" for event in self.detection_events),
                "disease_true_positive": self.true_positives_disease,
                "toxin_true_positive": self.true_positives_toxin,
            },
            "toxin_true_positive_dosed_agents_evaluation_only": [
                event.dosed_agents
                for event in self.detection_events
                if event.hazard_type == "toxin" and event.true_positive
            ],
            "toxin_true_positives_fewer_than_two_dosed_agents_evaluation_only": (
                self.toxin_true_positives_fewer_than_two_dosed_agents
            ),
            "total_detection_events": len(self.detection_events),
            "operational_metrics_daily": daily_operational,
            "background_metrics_daily": daily_background,
            **self._dilation_metrics(),
            **background,
            "broadcasts_per_occupied_zone_per_day": latest_day.get(
                "broadcasts_per_occupied_zone_per_day"
            ),
            "broadcasts_per_1000_agents_per_day": latest_day.get(
                "broadcasts_per_1000_agents_per_day"
            ),
            "fraction_occupied_zones_alarming": latest_day.get("fraction_occupied_zones_alarming"),
            "issued_broadcast_precision": issued_precision,
            "epsilon_per_agent_per_day": epsilon_per_agent_per_day,
            "n_agents": self.n_agents,
            "anomaly_threshold": self.anomaly_threshold,
            "aggregation_threshold": self.aggregation_threshold,
            "detector_mode": self.detector_mode,
            "sequential_reference_value": self.sequential_reference_value,
            "sequential_threshold": self.sequential_threshold,
            "sequential_clear_steps": self.sequential_clear_steps,
            "sequential_clear_fraction": self.sequential_clear_fraction,
            "sequential_residual_ewma_alpha": self.sequential_residual_ewma_alpha,
            "cardiac_detections": self.cardiac_detection_count(),
            "total_epsilon": self.epsilon_per_step[-1] if self.epsilon_per_step else 0.0,
            "response_mechanism": self.response_mechanism,
            "response_epsilon_basis": self.response_epsilon_basis,
            "response_epsilon_per_response": self.response_epsilon_per_response,
            "unaffected_positive_reply_probability": (self.unaffected_positive_reply_probability),
            "aggregate_count_epsilon": self.aggregate_count_epsilon,
            "aggregate_count_epsilon_per_release": self.aggregate_count_epsilon_per_release,
            "aggregate_count_composed_epsilon": self.aggregate_count_composed_epsilon,
            "aggregate_count_evidence_threshold": self.aggregate_count_evidence_threshold,
            "aggregate_count_minimum_releasable_count": (
                self.aggregate_count_minimum_releasable_count
            ),
            "aggregate_count_release_count": len(self.aggregate_count_releases),
            "aggregate_count_releases": self.aggregate_count_releases,
            "aggregate_count_no_cluster_releases": self.aggregate_count_no_cluster_releases,
            "aggregate_count_below_floor_releases": self.aggregate_count_below_floor_releases,
            "aggregate_count_evidence_releases": self.aggregate_count_evidence_releases,
            "geo_epsilon_basis": self.geo_epsilon_basis,
            "geo_epsilon_per_metre": self.geo_epsilon_per_metre,
            "total_broadcasts": self.total_queries_issued,
            "total_responses": self.total_responses,
            "disambiguation_queries_issued": self.disambiguation_queries_issued,
            "disambiguation_asks_suppressed_by_budget": (
                self.disambiguation_asks_suppressed_by_budget
            ),
            "disambiguation_acks": self.disambiguation_acks,
            "disambiguation_ack_release_count": self.disambiguation_ack_release_count,
            "disambiguation_devices_reached": self.disambiguation_devices_reached,
            "disambiguation_yes_answers": self.disambiguation_yes_answers,
            "disambiguation_no_answers": self.disambiguation_no_answers,
            "disambiguation_unanswered_expired": self.disambiguation_unanswered_expired,
            "disambiguation_unresolved_hypotheses": self.disambiguation_unresolved_hypotheses,
            "disambiguation_answer_epsilon": self.disambiguation_answer_epsilon,
            "disambiguation_ack_epsilon": self.disambiguation_ack_epsilon,
            "disambiguation_ack_epsilon_basis": self.disambiguation_ack_epsilon_basis,
            "disambiguation_well_founded_queries": self.disambiguation_well_founded_queries,
            "disambiguation_unfounded_queries": self.disambiguation_unfounded_queries,
            "disambiguation_unscored_queries": self.disambiguation_unscored_queries,
            "disambiguation_unfounded_ask_epsilon": (self.disambiguation_unfounded_ask_epsilon),
            "disambiguation_unscored_ask_epsilon": (self.disambiguation_unscored_ask_epsilon),
            "disambiguation_max_ask_epsilon_delta": (self.disambiguation_max_ask_epsilon_delta),
            "disambiguation_well_founded_by_hypothesis": dict(
                self.disambiguation_well_founded_by_hypothesis
            ),
            "disambiguation_unfounded_by_hypothesis": dict(
                self.disambiguation_unfounded_by_hypothesis
            ),
            "disambiguation_unscored_by_hypothesis": dict(
                self.disambiguation_unscored_by_hypothesis
            ),
            "disambiguation_precision": disambiguation_precision,
            "disambiguation_precision_by_hypothesis": precision_by_hypothesis,
            "confounder_contributions_by_cause": dict(self.confounder_contributions_by_cause),
            "confounder_agents_affected_by_cause": {
                cause: len(agents)
                for cause, agents in self.confounder_agents_affected_by_cause.items()
            },
            "heat_wave_active_steps": self.heat_wave_active_steps,
            "heat_wave_instances": dict(self.heat_wave_instances),
            "benign_overlap_detections": self.benign_overlap_detections,
            "benign_attributed_detections": self.benign_attributed_detections,
            "benign_misattributed_detections": self.benign_misattributed_detections,
            "benign_misattribution_rate": (
                self.benign_misattributed_detections
                / sum(not event.true_positive for event in self.detection_events)
                if any(not event.true_positive for event in self.detection_events)
                else 0.0
            ),
            "benign_misattributions_by_cause": dict(
                sorted(self.benign_misattributions_by_cause.items())
            ),
            "benign_coincident_true_positives": self.benign_coincident_true_positives,
            "target_detections": self.target_detections,
            "actionable_non_target_detections": self.actionable_non_target_detections,
            "explained_detections": self.explained_detections,
            "artifact_detections": self.artifact_detections,
            "unexplained_detections": self.unexplained_detections,
            "warranted_detections": self.warranted_detections,
            "artifact_detection_rate": (
                self.artifact_detections / len(self.detection_events)
                if self.detection_events
                else 0.0
            ),
            "unexplained_detection_rate": (
                self.unexplained_detections / len(self.detection_events)
                if self.detection_events
                else 0.0
            ),
            "sybil_false_alerts": self.sybil_false_alerts,
            "deanon_attempts": self.deanon_attempts,
            "deanon_successes": self.deanon_successes,
            "deanon_success_rate": (
                self.deanon_successes / self.deanon_attempts if self.deanon_attempts > 0 else 0.0
            ),
            "eclipse_tokens_dropped": self.eclipse_tokens_dropped,
            "correlation_evaluations": self.correlation_evaluations,
            "correlation_successes": self.correlation_successes,
            "correlation_success_rate": (
                self.correlation_successes / self.correlation_evaluations
                if self.correlation_evaluations > 0
                else 0.0
            ),
            "replay_tokens_injected": self.replay_tokens_injected,
            "replay_false_alerts": self.replay_false_alerts,
            "disease_onset_steps": dict(self.disease_onset_steps),
            "toxin_onset_steps": dict(self.toxin_onset_steps),
            "instance_true_positives": dict(self.instance_true_positives),
            "baseline_warmup_steps": self.baseline_warmup_steps,
            "warmup_step_count": self.warmup_step_count(),
            "baseline_maturation_minimum_history_days_evaluation_only": (
                self.baseline_maturation_minimum_history_days
            ),
            "baseline_maturation_maximum_history_days_evaluation_only": (
                self.baseline_maturation_maximum_history_days
            ),
            "baseline_maturation_cadence_steps_evaluation_only": (
                self.baseline_maturation_cadence_steps
            ),
            "baseline_maturation_device_count_evaluation_only": (
                self.baseline_maturation_device_count
            ),
            "baseline_maturation_history_days_min_evaluation_only": (
                min(self.baseline_maturation_history_days)
                if self.baseline_maturation_history_days
                else None
            ),
            "baseline_maturation_history_days_max_evaluation_only": (
                max(self.baseline_maturation_history_days)
                if self.baseline_maturation_history_days
                else None
            ),
            "baseline_maturation_history_days_mean_evaluation_only": (
                float(np.mean(self.baseline_maturation_history_days))
                if self.baseline_maturation_history_days
                else None
            ),
            "baseline_maturation_samples_per_device_min_evaluation_only": (
                min(self.baseline_maturation_sample_counts)
                if self.baseline_maturation_sample_counts
                else None
            ),
            "baseline_maturation_samples_per_device_max_evaluation_only": (
                max(self.baseline_maturation_sample_counts)
                if self.baseline_maturation_sample_counts
                else None
            ),
            "baseline_maturation_samples_per_device_mean_evaluation_only": (
                float(np.mean(self.baseline_maturation_sample_counts))
                if self.baseline_maturation_sample_counts
                else None
            ),
            "baseline_maturation_circadian_bins_min_evaluation_only": (
                min(self.baseline_maturation_circadian_bins)
                if self.baseline_maturation_circadian_bins
                else None
            ),
            "baseline_maturation_circadian_bins_max_evaluation_only": (
                max(self.baseline_maturation_circadian_bins)
                if self.baseline_maturation_circadian_bins
                else None
            ),
            "baseline_maturation_monthly_bins_min_evaluation_only": (
                min(self.baseline_maturation_monthly_bins)
                if self.baseline_maturation_monthly_bins
                else None
            ),
            "baseline_maturation_monthly_bins_max_evaluation_only": (
                max(self.baseline_maturation_monthly_bins)
                if self.baseline_maturation_monthly_bins
                else None
            ),
            **self._settlement_marker_summary(),
            "cause_attributed_detections": cause_counts,
            "cause_attribution_rates": cause_rates,
            "detection_power": self.detection_power.summary(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert step records to pandas DataFrame."""
        return pd.DataFrame(self.step_records)

    def export_csv(self, path: str | Path) -> None:
        """Export step-level metrics to CSV."""
        import io

        buffer = io.StringIO()
        self.to_dataframe().to_csv(buffer, index=False)
        write_text_file(path, buffer.getvalue())

    def plot_metrics(self, output_dir: str | Path) -> None:
        """Generate diagnostic plots for simulation evaluation.

        Creates:
        1. SEIR curve (S, E, I, R over time)
        2. Detection timeline (hazard onset vs. system detection)
        3. Epsilon budget over time
        4. Privacy protocol activity (tokens, broadcasts, responses)
        """
        import matplotlib.pyplot as plt

        output_dir = ensure_directory(output_dir)
        df = self.to_dataframe()

        if df.empty:
            return

        # 1. SEIR Curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["time_hours"], df["susceptible"], label="Susceptible", alpha=0.7)
        ax.plot(df["time_hours"], df["exposed"], label="Exposed", alpha=0.7)
        ax.plot(df["time_hours"], df["infectious"], label="Infectious", alpha=0.7)
        ax.plot(df["time_hours"], df["recovered"], label="Recovered", alpha=0.7)
        ax.set_xlabel(_TIME_HOURS_LABEL)
        ax.set_ylabel("Agent Count")
        ax.set_title("SEIR Dynamics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_figure(
            fig,
            resolve_under_base(output_dir, "seir_curve.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # 2. Detection Timeline
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["time_hours"], df["anomalies_detected"], label="Anomalies Detected")
        ax.plot(df["time_hours"], df["plume_exposed"], label="Plume Exposed", alpha=0.7)
        if self.disease_onset_step is not None:
            ax.axvline(
                self.disease_onset_step * 5 / 60,
                color="red",
                linestyle="--",
                label="Disease Onset",
            )
        if self.disease_detection_step is not None:
            ax.axvline(
                self.disease_detection_step * 5 / 60,
                color="red",
                linestyle="-",
                label="Disease Detected",
            )
        if self.toxin_onset_step is not None:
            ax.axvline(
                self.toxin_onset_step * 5 / 60,
                color="orange",
                linestyle="--",
                label="Toxin Onset",
            )
        if self.toxin_detection_step is not None:
            ax.axvline(
                self.toxin_detection_step * 5 / 60,
                color="orange",
                linestyle="-",
                label="Toxin Detected",
            )
        ax.set_xlabel(_TIME_HOURS_LABEL)
        ax.set_ylabel("Count")
        ax.set_title("Hazard Detection Timeline")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        save_figure(
            fig,
            resolve_under_base(output_dir, "detection_timeline.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # 3. Epsilon Budget
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["time_hours"], df["cumulative_epsilon"], color="purple")
        ax.set_xlabel(_TIME_HOURS_LABEL)
        ax.set_ylabel("Cumulative ε")
        ax.set_title("Privacy Budget Expenditure (Adaptive Composition)")
        ax.grid(True, alpha=0.3)
        save_figure(
            fig,
            resolve_under_base(output_dir, "epsilon_budget.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # 4. System activity
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["time_hours"], df["tokens_submitted"], label="Tokens Submitted")
        ax.plot(df["time_hours"], df["broadcasts_issued"], label="Broadcasts Issued")
        ax.plot(df["time_hours"], df["responses_received"], label="Responses")
        ax.set_xlabel(_TIME_HOURS_LABEL)
        ax.set_ylabel("Count")
        ax.set_title("Privacy Protocol Activity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_figure(
            fig,
            resolve_under_base(output_dir, "protocol_activity.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
