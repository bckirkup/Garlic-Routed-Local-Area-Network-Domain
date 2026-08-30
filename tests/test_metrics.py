"""Tests for episode-granular metrics counting in MetricsCollector."""

from __future__ import annotations

import pytest

from garland.metrics import DetectionEvent, MetricsCollector
from garland.perturbations import PerturbationCause
from garland.privacy import AnomalyType


def _disease_tp(step: int = 0) -> DetectionEvent:
    return DetectionEvent(
        step=step,
        hazard_type="disease",
        anomaly_type=AnomalyType.FEBRILE,
        zone_id=0,
        true_positive=True,
        agents_affected=1,
    )


def _disease_fp(step: int = 0) -> DetectionEvent:
    return DetectionEvent(
        step=step,
        hazard_type="disease",
        anomaly_type=AnomalyType.FEBRILE,
        zone_id=0,
        true_positive=False,
        agents_affected=1,
    )


def _toxin_tp(step: int = 0, attributed: bool | None = None) -> DetectionEvent:
    return DetectionEvent(
        step=step,
        hazard_type="toxin",
        anomaly_type=AnomalyType.RESPIRATORY,
        zone_id=0,
        true_positive=True,
        agents_affected=1,
        attributed=attributed,
    )


class TestEpisodeFalseNegatives:
    """FNR should count at most one FN per undetected hazard episode."""

    def test_undetected_episode_counts_one_fn_over_many_steps(self):
        metrics = MetricsCollector()
        metrics.update_hazard_episode("disease", True, False, False)
        for _ in range(49):
            metrics.update_hazard_episode("disease", True, False, False)
        metrics.finalize_hazard_episodes()
        assert metrics.false_negatives_disease == 1

    def test_detected_episode_counts_zero_fn(self):
        metrics = MetricsCollector()
        metrics.update_hazard_episode("disease", True, False, False)
        metrics.record_detection(_disease_tp(step=1))
        metrics.update_hazard_episode("disease", True, True, False)
        for _ in range(10):
            metrics.update_hazard_episode("disease", True, False, False)
        metrics.finalize_hazard_episodes()
        assert metrics.false_negatives_disease == 0
        assert metrics.true_positives_disease == 1

    def test_episode_reopen_after_inactive_counts_separate_fn(self):
        metrics = MetricsCollector()
        metrics.update_hazard_episode("disease", True, False, False)
        metrics.update_hazard_episode("disease", False, False, False)
        metrics.update_hazard_episode("disease", True, False, False)
        metrics.finalize_hazard_episodes()
        assert metrics.false_negatives_disease == 2

    def test_fnr_formula_with_episode_counting(self):
        metrics = MetricsCollector()
        metrics.record_missed_detection("disease")
        metrics.record_detection(_disease_tp())
        assert metrics.false_negative_rate_disease() == pytest.approx(0.5)


def test_dilation_metrics_record_means_percentiles_and_k_coverage():
    metrics = MetricsCollector()
    metrics.record_dilation(
        dilated_cell_count=1,
        resident_population=100,
        estimated_respondent_population=40,
        true_respondent_population=45,
        k_min=50,
    )
    metrics.record_dilation(
        dilated_cell_count=3,
        resident_population=200,
        estimated_respondent_population=60,
        true_respondent_population=55,
        k_min=50,
    )
    summary = metrics.summary()
    assert summary["dilation_broadcasts"] == 2
    assert summary["dilated_cells_mean"] == pytest.approx(2.0)
    assert summary["dilated_cells_p90"] == pytest.approx(2.8)
    assert summary["resident_population_mean"] == pytest.approx(150.0)
    assert summary["fraction_true_respondents_meeting_k"] == pytest.approx(0.5)


def test_dilation_metrics_record_suppressed_triggers_without_issued_queries():
    metrics = MetricsCollector()
    metrics.record_dilation_suppressed(estimated_respondent_population=12, k_min=50)
    summary = metrics.summary()
    assert summary["dilation_broadcasts"] == 0
    assert summary["dilation_suppressed_for_insufficient_anonymity"] == 1
    assert summary["dilation_suppression_rate"] == pytest.approx(1.0)
    assert summary["suppressed_estimated_respondent_population_mean"] == pytest.approx(12.0)
    assert summary["fraction_true_respondents_meeting_k"] is None


def test_dilation_metrics_record_under_k_release_and_epsilon():
    metrics = MetricsCollector()
    metrics.record_dilation(
        dilated_cell_count=2,
        resident_population=100,
        estimated_respondent_population=60,
        true_respondent_population=40,
        k_min=50,
        step=12,
        responding_devices=40,
        release_suppressed=True,
        response_epsilon_burned=0.25,
    )
    summary = metrics.summary()
    assert metrics.dilation_records[0]["step"] == 12
    assert summary["dilation_release_suppressed_for_insufficient_anonymity"] == 1
    assert summary["dilation_release_suppression_rate"] == pytest.approx(1.0)
    assert summary["dilation_release_suppressed_epsilon"] == pytest.approx(0.25)
    assert summary["dilation_release_suppressed_epsilon_share"] == pytest.approx(1.0)


class TestEpisodeTrueNegatives:
    """FPR denominator TN should count at most one TN per no-hazard episode."""

    def test_no_hazard_episode_counts_one_tn_over_many_steps(self):
        metrics = MetricsCollector()
        for _ in range(50):
            metrics.update_hazard_episode("disease", False, False, False)
        metrics.finalize_hazard_episodes()
        assert metrics.true_negatives_disease == 1

    def test_false_positive_suppresses_tn_for_episode(self):
        metrics = MetricsCollector()
        metrics.update_hazard_episode("disease", False, False, True)
        for _ in range(10):
            metrics.update_hazard_episode("disease", False, False, False)
        metrics.finalize_hazard_episodes()
        assert metrics.true_negatives_disease == 0

    def test_hazard_transition_closes_no_hazard_episode_with_tn(self):
        metrics = MetricsCollector()
        for _ in range(5):
            metrics.update_hazard_episode("disease", False, False, False)
        metrics.update_hazard_episode("disease", True, False, False)
        assert metrics.true_negatives_disease == 1


class TestSummaryWiring:
    """Summary counters updated during record_step."""

    def test_record_step_accumulates_broadcasts_and_responses(self):
        metrics = MetricsCollector()
        metrics.record_step(
            step=0,
            seir_counts={"S": 10, "E": 0, "I": 0, "R": 0},
            plume_exposed=0,
            anomalies_detected=0,
            tokens_submitted=5,
            broadcasts_issued=2,
            responses_received=3,
            cumulative_epsilon=0.1,
        )
        metrics.record_step(
            step=1,
            seir_counts={"S": 10, "E": 0, "I": 0, "R": 0},
            plume_exposed=0,
            anomalies_detected=0,
            tokens_submitted=4,
            broadcasts_issued=1,
            responses_received=2,
            cumulative_epsilon=0.2,
        )
        assert metrics.total_queries_issued == 3
        assert metrics.total_responses == 5


class TestAttributedDetectionMetrics:
    def test_benign_overlap_attribution_and_misattribution_counters(self):
        metrics = MetricsCollector()
        metrics.record_detection(
            DetectionEvent(
                step=0,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=False,
                agents_affected=2,
                benign_instance_id="ili_0",
                benign_cause=PerturbationCause.BACKGROUND_ILI,
                benign_attributed=True,
                causes=frozenset(),
            )
        )
        metrics.record_detection(
            DetectionEvent(
                step=1,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=True,
                agents_affected=2,
                benign_instance_id="heat_0",
                benign_cause=PerturbationCause.HEAT_WAVE,
                benign_attributed=True,
            )
        )
        summary = metrics.summary()
        assert summary["benign_overlap_detections"] == 2
        assert summary["benign_attributed_detections"] == 2
        assert summary["benign_misattributed_detections"] == 1
        assert summary["benign_misattribution_rate"] == pytest.approx(1.0)
        assert summary["benign_coincident_true_positives"] == 1
        assert summary["benign_misattributions_by_cause"] == {"background_ili": 1}
        assert (
            sum(summary["benign_misattributions_by_cause"].values())
            == (summary["benign_misattributed_detections"])
        )
        assert (
            summary["benign_attributed_detections"]
            <= summary["benign_overlap_detections"]
            <= len(metrics.detection_events)
        )
        assert summary["benign_misattributed_detections"] <= summary["benign_attributed_detections"]

    def test_warrant_classes_conserve_detection_events(self):
        metrics = MetricsCollector()
        events = [
            DetectionEvent(
                step=0,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=True,
                agents_affected=1,
            ),
            DetectionEvent(
                step=1,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=False,
                agents_affected=1,
                benign_cause=PerturbationCause.HEAT_WAVE,
            ),
            DetectionEvent(
                step=2,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=False,
                agents_affected=1,
                benign_cause=PerturbationCause.EXERCISE,
            ),
            DetectionEvent(
                step=3,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=False,
                agents_affected=1,
                causes=frozenset({PerturbationCause.SENSOR_ARTIFACT}),
            ),
            DetectionEvent(
                step=4,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=False,
                agents_affected=1,
            ),
        ]
        for event in events:
            metrics.record_detection(event)
        summary = metrics.summary()
        classes = (
            summary["target_detections"],
            summary["actionable_non_target_detections"],
            summary["explained_detections"],
            summary["artifact_detections"],
            summary["unexplained_detections"],
        )
        assert sum(classes) == len(events)
        assert summary["warranted_detections"] == 2
        assert 0.0 <= summary["artifact_detection_rate"] <= 1.0
        assert 0.0 <= summary["unexplained_detection_rate"] <= 1.0

    def test_attributed_and_coincidental_counts_and_latency(self):
        metrics = MetricsCollector(toxin_onset_step=10)
        metrics.record_detection(_toxin_tp(step=12, attributed=False))
        metrics.record_detection(_toxin_tp(step=15, attributed=True))
        summary = metrics.summary()

        assert "attributed_detection" not in summary
        assert summary["attributed_toxin_detections"] == 1
        assert summary["coincidental_toxin_detections"] == 1
        assert summary["detection_event_counts"]["toxin_true_positive"] == 2
        assert summary["attributed_toxin_latency_steps"] == pytest.approx(5.0)
        assert summary["coincidental_fraction_toxin"] == pytest.approx(0.5)

    def test_no_attributed_evidence_uses_none_not_zero(self):
        metrics = MetricsCollector(toxin_onset_step=10)
        metrics.record_detection(_toxin_tp(step=12, attributed=False))
        summary = metrics.summary()

        assert summary["attributed_toxin_latency_steps"] is None
        assert summary["coincidental_fraction_toxin"] == pytest.approx(1.0)

    def test_no_hazard_detections_use_none_semantics(self):
        summary = MetricsCollector().summary()
        assert summary["attributed_disease_latency_steps"] is None
        assert summary["attributed_toxin_latency_steps"] is None
        assert summary["coincidental_fraction_disease"] is None
        assert summary["coincidental_fraction_toxin"] is None

    def test_affected_token_fragmentation_has_golden_values(self):
        metrics = MetricsCollector()
        metrics.record_affected_agent_token("toxin", AnomalyType.RESPIRATORY, 2)
        metrics.record_affected_agent_token("toxin", AnomalyType.RESPIRATORY, 3)
        metrics.record_affected_agent_token("toxin", AnomalyType.CARDIAC, 1)

        summary = metrics.summary()
        assert "affected_agent_tokens" not in summary
        assert summary["affected_agent_token_counts"]["toxin"] == {
            "respiratory": 2,
            "cardiac": 1,
        }
        assert summary["largest_affected_agent_group"]["toxin"] == 3


class TestAttackMetrics:
    """Attack-related summary fields should update during simulation."""

    def test_record_sybil_false_alert(self):
        metrics = MetricsCollector()
        metrics.record_sybil_false_alert()
        metrics.record_sybil_false_alert(2)
        assert metrics.sybil_false_alerts == 3
        assert metrics.summary()["sybil_false_alerts"] == 3

    def test_sync_attack_metrics(self):
        from garland.attacks import AttackOrchestrator

        metrics = MetricsCollector()
        orchestrator = AttackOrchestrator()
        orchestrator.deanon_attempts = 4
        orchestrator.deanon_successes = 1
        orchestrator.eclipse.dropped_count = 7
        orchestrator.correlation_evaluations = 2
        orchestrator.correlation_successes = 1
        orchestrator.replay.replay_inject_count = 15
        orchestrator.replay_false_alerts = 3
        metrics.sync_attack_metrics(orchestrator)
        summary = metrics.summary()
        assert metrics.deanon_attempts == 4
        assert metrics.deanon_successes == 1
        assert summary["deanon_success_rate"] == pytest.approx(0.25)
        assert summary["eclipse_tokens_dropped"] == 7
        assert summary["correlation_evaluations"] == 2
        assert summary["correlation_success_rate"] == pytest.approx(0.5)
        assert summary["replay_tokens_injected"] == 15
        assert summary["replay_false_alerts"] == 3


class TestCardiacMetrics:
    """Cardiac detections should contribute to discrimination scoring."""

    def test_cardiac_toxin_event_counts_as_discriminated(self):
        metrics = MetricsCollector()
        metrics.record_detection(
            DetectionEvent(
                step=0,
                hazard_type="toxin",
                anomaly_type=AnomalyType.CARDIAC,
                zone_id=0,
                true_positive=True,
                agents_affected=1,
            )
        )
        assert metrics.discrimination_score() is None
        assert metrics.cardiac_detection_count() == 1

    def test_cardiac_disease_event_counts_as_discriminated(self):
        metrics = MetricsCollector()
        metrics.record_detection(
            DetectionEvent(
                step=0,
                hazard_type="disease",
                anomaly_type=AnomalyType.CARDIAC,
                zone_id=0,
                true_positive=True,
                agents_affected=1,
            )
        )
        assert metrics.discrimination_score() is None

    def test_missing_hazard_class_is_not_perfect_discrimination(self):
        metrics = MetricsCollector()
        metrics.record_detection(_disease_tp())
        assert metrics.discrimination_score() is None


class TestAggregateCountMetrics:
    def test_summary_distinguishes_no_cluster_and_below_floor(self):
        metrics = MetricsCollector()
        metrics.configure_privacy_accounting(
            response_mechanism="aggregate_noisy_count",
            response_basis="mechanism",
            response_epsilon=0.0,
            unaffected_positive_probability=None,
            aggregate_count_epsilon_per_release=1.0,
            aggregate_count_evidence_threshold=3,
            aggregate_count_minimum_releasable_count=4,
            geo_epsilon_per_metre=0.0,
            geo_basis="separate",
            ack_basis="configured",
        )
        for query_id, released_count in enumerate((0, 2, 4)):
            metrics.record_aggregate_count_release(
                query_id=query_id,
                released_count=released_count,
                true_count=released_count,
                population=20,
                epsilon=1.0,
                composed_epsilon=float(query_id + 1),
                evidence_threshold=3,
            )

        summary = metrics.summary()
        assert summary["aggregate_count_no_cluster_releases"] == 1
        assert summary["aggregate_count_below_floor_releases"] == 1
        assert summary["aggregate_count_evidence_releases"] == 1
        assert summary["aggregate_count_minimum_releasable_count"] == 4
        assert summary["aggregate_count_epsilon_per_release"] == pytest.approx(1.0)
        assert summary["aggregate_count_composed_epsilon"] == pytest.approx(3.0)
        assert [row["outcome"] for row in summary["aggregate_count_releases"]] == [
            "no_cluster",
            "cluster_below_floor",
            "evidence",
        ]


class TestPlotMetrics:
    """plot_metrics should write diagnostic PNGs for non-empty runs."""

    def test_plot_metrics_writes_pngs(self, tmp_path):
        metrics = MetricsCollector()
        for step in range(5):
            metrics.record_step(
                step=step,
                seir_counts={"S": 90, "E": 5, "I": 3, "R": 2},
                plume_exposed=0,
                anomalies_detected=step,
                tokens_submitted=step,
                broadcasts_issued=1,
                responses_received=2,
                cumulative_epsilon=0.1 * step,
            )
        metrics.plot_metrics(tmp_path)
        assert (tmp_path / "seir_curve.png").exists()
        assert (tmp_path / "detection_timeline.png").exists()
        assert (tmp_path / "epsilon_budget.png").exists()
        assert (tmp_path / "protocol_activity.png").exists()


class TestHazardDetectionsDaily:
    """summary should bucket hazard true positives and attribution by day."""

    def test_daily_buckets_split_attributed_and_coincidental(self):
        metrics = MetricsCollector()
        metrics.record_detection(
            DetectionEvent(
                step=10,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=True,
                agents_affected=1,
                attributed=True,
            )
        )
        metrics.record_detection(
            DetectionEvent(
                step=300,
                hazard_type="disease",
                anomaly_type=AnomalyType.FEBRILE,
                zone_id=0,
                true_positive=True,
                agents_affected=1,
                attributed=False,
            )
        )
        metrics.record_detection(_toxin_tp(step=300, attributed=True))
        metrics.record_detection(_disease_fp(step=300))
        daily = metrics.summary()["hazard_detections_daily"]
        assert daily == {
            "0": {"disease": {"true_positives": 1, "attributed": 1, "coincidental": 0}},
            "1": {
                "disease": {"true_positives": 1, "attributed": 0, "coincidental": 1},
                "toxin": {"true_positives": 1, "attributed": 1, "coincidental": 0},
            },
        }
