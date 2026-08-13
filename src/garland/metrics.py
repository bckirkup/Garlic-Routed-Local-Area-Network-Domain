"""Evaluation metrics for the GARLAND epidemiological security testbed.

Tracks and outputs:
- Time-to-Detection for both hazard types (disease and toxin)
- False Positive / False Negative rates
- Hazard discrimination (decoupling toxin from infection)
- Cumulative Privacy Budget (Epsilon) under adaptive composition
- Per-step CSV output for downstream analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from garland.paths import (
    ensure_directory,
    resolve_under_base,
    save_figure,
    write_text_file,
)
from garland.perturbations import PerturbationCause
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
    causes: frozenset[PerturbationCause] = frozenset()


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
    detector_mode: str | None = None
    sequential_reference_value: float | None = None
    sequential_threshold: float | None = None
    sequential_clear_steps: int | None = None
    sequential_clear_fraction: float | None = None
    sequential_residual_ewma_alpha: float | None = None
    _daily_occupied_zones: dict[int, set[int]] = field(default_factory=dict)
    _daily_alarming_zones: dict[int, set[int]] = field(default_factory=dict)
    cause_attributed_detections: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"disease": {}, "toxin": {}}
    )
    _step_cause_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"disease": {}, "toxin": {}}
    )

    @staticmethod
    def _cause_bucket(
        hazard_type: str, cause: PerturbationCause
    ) -> str:
        return "hazard" if cause.value == hazard_type else cause.value

    def _record_cause_counts(self, event: DetectionEvent) -> None:
        causes = event.causes
        if not causes:
            buckets = {"none"}
        else:
            buckets = {
                self._cause_bucket(event.hazard_type, cause) for cause in causes
            }
        for bucket in buckets:
            self.cause_attributed_detections[event.hazard_type][bucket] = (
                self.cause_attributed_detections[event.hazard_type].get(bucket, 0) + 1
            )
            self._step_cause_counts[event.hazard_type][bucket] = (
                self._step_cause_counts[event.hazard_type].get(bucket, 0) + 1
            )

    def record_baseline_warmup_config(self, steps: int) -> None:
        """Store configured baseline warm-up length for summary output."""
        self.baseline_warmup_steps = steps

    def record_population_config(self, n_agents: int) -> None:
        """Store population size for per-agent operational metrics."""
        self.n_agents = n_agents

    def record_anomaly_threshold_config(self, threshold: float) -> None:
        """Store the configured anomaly threshold for reproducibility."""
        self.anomaly_threshold = threshold

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
        self._record_cause_counts(event)
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
        occupied_zone_ids: set[int] | None = None,
        alarming_zone_ids: set[int] | None = None,
        cause_counts: dict[str, dict[str, int]] | None = None,
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
                "mean_battery_level": mean_battery_level,
                "baseline_warmup_active": baseline_warmup_active,
                "wearables_in_warmup": wearables_in_warmup,
                "occupied_zones": len(occupied_zone_ids or set()),
                "alarming_zones": len(alarming_zone_ids or set()),
            }
        for hazard_type in ("disease", "toxin"):
            for cause in PerturbationCause:
                bucket = self._cause_bucket(hazard_type, cause)
                record[f"{hazard_type}_cause_{bucket}"] = (
                    (cause_counts or self._step_cause_counts)
                    .get(hazard_type, {})
                    .get(bucket, 0)
                )
            record[f"{hazard_type}_cause_none"] = (
                (cause_counts or self._step_cause_counts)
                .get(hazard_type, {})
                .get("none", 0)
            )
        self.step_records.append(record)
        self._step_cause_counts = {"disease": {}, "toxin": {}}
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

    def summary(self) -> dict:
        """Generate summary metrics dictionary."""
        ttd_disease = self.time_to_detection_disease()
        ttd_toxin = self.time_to_detection_toxin()
        attributed_ttd_disease = self.attributed_time_to_detection_disease()
        attributed_ttd_toxin = self.attributed_time_to_detection_toxin()
        daily_operational = self._operational_daily_metrics()
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
        cause_buckets = ("hazard", "none") + tuple(
            cause.value for cause in PerturbationCause
        )
        cause_counts = {
            hazard_type: {
                bucket: counts.get(bucket, 0)
                for bucket in cause_buckets
            }
            for hazard_type, counts in self.cause_attributed_detections.items()
        }
        cause_rates: dict[str, dict[str, float | None]] = {}
        for hazard_type, counts in cause_counts.items():
            denominator = sum(counts.values())
            cause_rates[hazard_type] = {
                bucket: (count / denominator if denominator else None)
                for bucket, count in counts.items()
            }
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
            "cause_attributed_detections": cause_counts,
            "cause_attributed_broadcasts": cause_counts,
            "cause_attribution_rates": cause_rates,
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
