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
from pathlib import Path

import pandas as pd

from garland.paths import (
    ensure_directory,
    resolve_under_base,
    save_figure,
    write_text_file,
)
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
    hazard_instance_id: str | None = None
    attributed: bool | None = None


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
    emission_bucket_stats: dict[tuple[AnomalyType, str], list[float]] = field(
        default_factory=dict
    )
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
    window_bucket_stats: dict[tuple[AnomalyType, str], list[float]] = field(
        default_factory=dict
    )
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

    # System detection times
    disease_detection_step: int | None = None
    toxin_detection_step: int | None = None

    # Per-step tracking
    step_records: list[dict] = field(default_factory=list)
    detection_events: list[DetectionEvent] = field(default_factory=list)

    # Confusion matrix accumulators
    true_positives_disease: int = 0
    false_positives_disease: int = 0
    true_negatives_disease: int = 0
    false_negatives_disease: int = 0
    true_positives_toxin: int = 0
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
    _background_open_groups: dict[tuple[int, AnomalyType], list[int]] = field(
        default_factory=dict
    )
    _background_emission_n: dict[AnomalyType, int] = field(default_factory=dict)
    _background_emission_sum_c: dict[AnomalyType, int] = field(default_factory=dict)
    _background_emission_sum_e: dict[AnomalyType, int] = field(default_factory=dict)
    _background_emission_sum_c2_over_e: dict[AnomalyType, float] = field(
        default_factory=dict
    )
    _background_emission_sum_c2: dict[AnomalyType, int] = field(default_factory=dict)
    _background_emission_e_hist: dict[AnomalyType, dict[int, int]] = field(
        default_factory=dict
    )
    _background_emission_bucket_stats: dict[
        tuple[AnomalyType, str], list[float]
    ] = field(default_factory=dict)
    _background_emission_observed: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_history: dict[
        tuple[int, AnomalyType], deque[tuple[int, int]]
    ] = field(default_factory=dict)
    _background_window_sum_c: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_sum_e: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_sum_c2_over_e: dict[AnomalyType, float] = field(
        default_factory=dict
    )
    _background_window_sum_c2: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_sum_c_fold: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_sum_e_fold: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_n: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_e_hist: dict[AnomalyType, dict[int, int]] = field(
        default_factory=dict
    )
    _background_window_bucket_stats: dict[
        tuple[AnomalyType, str], list[float]
    ] = field(default_factory=dict)
    _background_window_observed: dict[AnomalyType, int] = field(default_factory=dict)
    _background_window_triggered: dict[tuple[int, AnomalyType], bool] = field(
        default_factory=dict
    )
    background_burn_in_steps: int = 0
    _background_settled_tokens: int = 0
    _background_settled_eligible: int = 0
    _background_settled_population_n: int = 0
    _background_settled_population_sum: int = 0
    _background_settled_population_sum_sq: int = 0
    _background_settled_type_tokens: dict[AnomalyType, int] = field(default_factory=dict)
    _background_settled_type_eligible: dict[AnomalyType, int] = field(
        default_factory=dict
    )
    _background_settled: _BackgroundFoldState = field(
        default_factory=_BackgroundFoldState
    )
    _background_assessment_finalized: dict[str, object] | None = None

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
        day = step // 288
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
        if step >= self.background_burn_in_steps:
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

    def _finalize_background_bin(self) -> None:
        """Fold one emission bin and one aggregation-window observation."""
        if self._background_open_time_bin is None:
            return
        current = self._background_open_groups
        keys = set(self._background_window_history) | set(current)
        for anomaly_type in AnomalyType:
            for zone_id, group_type in keys:
                if group_type != anomaly_type:
                    continue
                count, eligible = current.get((zone_id, anomaly_type), [0, 0])
                self._increment_group_fold(
                    anomaly_type=anomaly_type,
                    count=count,
                    eligible=eligible,
                    n=self._background_emission_n,
                    sum_c=self._background_emission_sum_c,
                    sum_e=self._background_emission_sum_e,
                    sum_c2_over_e=self._background_emission_sum_c2_over_e,
                    sum_c2=self._background_emission_sum_c2,
                    e_hist=self._background_emission_e_hist,
                    bucket_stats=self._background_emission_bucket_stats,
                )
                if (
                    self.aggregation_threshold is not None
                    and count >= self.aggregation_threshold
                ):
                    self._background_emission_observed[anomaly_type] = (
                        self._background_emission_observed.get(anomaly_type, 0) + 1
                    )
                history = self._background_window_history.setdefault(
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
                    n=self._background_window_n,
                    sum_c=self._background_window_sum_c_fold,
                    sum_e=self._background_window_sum_e_fold,
                    sum_c2_over_e=self._background_window_sum_c2_over_e,
                    sum_c2=self._background_window_sum_c2,
                    e_hist=self._background_window_e_hist,
                    bucket_stats=self._background_window_bucket_stats,
                )
                threshold = self.aggregation_threshold
                if threshold is not None and window_count >= threshold:
                    self._background_window_observed[anomaly_type] = (
                        self._background_window_observed.get(anomaly_type, 0) + 1
                    )
                    history.clear()
                    history.append((count, eligible))
                elif not any(history):
                    self._background_window_history.pop((zone_id, anomaly_type), None)
        self._background_open_groups = {}
        self._background_open_time_bin = None

    def _finalize_background_state(self, state: _BackgroundFoldState) -> None:
        if state.open_time_bin is None:
            return
        current = state.open_groups
        keys = set(state.window_history) | set(current)
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
            if (
                self.aggregation_threshold is not None
                and count >= self.aggregation_threshold
            ):
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
                state.window_observed[anomaly_type] = (
                    state.window_observed.get(anomaly_type, 0) + 1
                )
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
            "background_agent_zero_fraction": sum(count == 0 for count in counts)
            / len(counts),
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
            valid_types = [
                anomaly_type for anomaly_type in AnomalyType if rates[anomaly_type] > 0
            ]
            n_groups = sum(n_by_type.get(anomaly_type, 0) for anomaly_type in valid_types)
            sum_counts = sum(
                sum_c_by_type.get(anomaly_type, 0) for anomaly_type in valid_types
            )
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
            for (anomaly_type, label), (bucket_n, bucket_sum, bucket_sum_sq) in (
                bucket_stats_by_type.items()
            ):
                mean = bucket_sum / bucket_n
                bucket_results[f"{anomaly_type.value}:{label}"] = {
                    "n_groups": int(bucket_n),
                    "sum_counts": int(bucket_sum),
                    "variance_to_mean": (
                        (bucket_sum_sq / bucket_n - mean * mean) / mean
                        if mean
                        else None
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
                    self._poisson_tail(rates[anomaly_type] * eligible, threshold)
                    * count
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
            "background_emission_pearson_dispersion": emission[
                "pearson_dispersion"
            ],
            "background_emission_occupancy_buckets": emission["occupancy_buckets"],
            "background_emission_observed_at_threshold_fraction": emission[
                "groups_observed_at_threshold_fraction"
            ],
            "background_emission_poisson_tail_fraction": emission[
                "groups_poisson_tail_fraction"
            ],
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
            "background_window_poisson_tail_fraction": window[
                "groups_poisson_tail_fraction"
            ],
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
            "background_groups_poisson_tail_fraction": emission[
                "groups_poisson_tail_fraction"
            ],
            "background_window_length_bins": self.aggregation_window_bins,
            "background_settled_rate_by_anomaly_type": {
                anomaly_type.value: settled_rates[anomaly_type]
                for anomaly_type in AnomalyType
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
            "background_settled_emission_group_sum_counts": settled_emission[
                "group_sum_counts"
            ],
            "background_settled_emission_group_mean_lambda": settled_emission[
                "group_mean_lambda"
            ],
            "background_settled_emission_pearson_dispersion": settled_emission[
                "pearson_dispersion"
            ],
            "background_settled_emission_occupancy_buckets": settled_emission[
                "occupancy_buckets"
            ],
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
            "background_settled_window_group_sum_counts": settled_window[
                "group_sum_counts"
            ],
            "background_settled_window_group_mean_lambda": settled_window[
                "group_mean_lambda"
            ],
            "background_settled_window_pearson_dispersion": settled_window[
                "pearson_dispersion"
            ],
            "background_settled_window_occupancy_buckets": settled_window[
                "occupancy_buckets"
            ],
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

    def record_baseline_warmup_config(self, steps: int) -> None:
        """Store configured baseline warm-up length for summary output."""
        self.baseline_warmup_steps = steps

    def record_background_burn_in_config(self, steps: int) -> None:
        """Store the measurement-only background burn-in length."""
        self.background_burn_in_steps = steps

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
            self.instance_true_positives[key] = (
                self.instance_true_positives.get(key, 0) + 1
            )

    def _record_false_positive(self, event: DetectionEvent) -> None:
        if event.hazard_type == "disease":
            self.false_positives_disease += 1
        elif event.hazard_type == "toxin":
            self.false_positives_toxin += 1

    def record_detection(self, event: DetectionEvent) -> None:
        """Record a system detection event and update confusion matrix."""
        self.detection_events.append(event)
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
        occupied_zone_ids: set[int] | None = None,
        alarming_zone_ids: set[int] | None = None,
    ) -> None:
        """Record per-step metrics for CSV output."""
        self.step_records.append(
            {
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
                "mean_battery_level": mean_battery_level,
                "baseline_warmup_active": baseline_warmup_active,
                "wearables_in_warmup": wearables_in_warmup,
                "background_tokens": background_tokens,
                "background_eligible_wearables": background_eligible_wearables,
                "background_rate": background_rate,
                "occupied_zones": len(occupied_zone_ids or set()),
                "alarming_zones": len(alarming_zone_ids or set()),
            }
        )
        day = step // 288
        self._daily_occupied_zones.setdefault(day, set()).update(occupied_zone_ids or set())
        self._daily_alarming_zones.setdefault(day, set()).update(alarming_zone_ids or set())
        self.epsilon_per_step.append(cumulative_epsilon)
        self.total_queries_issued += broadcasts_issued
        self.total_responses += responses_received

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
        if (
            self.disease_onset_step is None
            or self.attributed_disease_detection_step is None
        ):
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
            day = int(record["step"]) // 288
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
                "fraction_occupied_zones_alarming": (
                    alarming / occupied if occupied else None
                ),
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
        elapsed_days = len(self.step_records) / 288
        epsilon_per_agent_per_day = (
            (self.epsilon_per_step[-1] if self.epsilon_per_step else 0.0)
            / self.n_agents
            / elapsed_days
            if self.n_agents and elapsed_days
            else None
        )
        complete_day_count = len(self.step_records) // 288
        latest_complete_day = str(complete_day_count - 1) if complete_day_count else None
        latest_day = (
            daily_operational.get(latest_complete_day, {})
            if latest_complete_day is not None
            else {}
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
            "operational_metrics_daily": daily_operational,
            "background_metrics_daily": daily_background,
            **background,
            "broadcasts_per_occupied_zone_per_day": latest_day.get(
                "broadcasts_per_occupied_zone_per_day"
            ),
            "broadcasts_per_1000_agents_per_day": latest_day.get(
                "broadcasts_per_1000_agents_per_day"
            ),
            "fraction_occupied_zones_alarming": latest_day.get(
                "fraction_occupied_zones_alarming"
            ),
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
            "total_broadcasts": self.total_queries_issued,
            "total_responses": self.total_responses,
            "sybil_false_alerts": self.sybil_false_alerts,
            "deanon_attempts": self.deanon_attempts,
            "deanon_successes": self.deanon_successes,
            "deanon_success_rate": (
                self.deanon_successes / self.deanon_attempts
                if self.deanon_attempts > 0
                else 0.0
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
