"""Tests for episode-granular metrics counting in MetricsCollector."""

from __future__ import annotations

from garland.metrics import DetectionEvent, MetricsCollector
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
        assert metrics.false_negative_rate_disease() == 0.5


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
        summary = metrics.summary()
        assert metrics.total_queries_issued == 3
        assert metrics.total_responses == 5
        assert summary["total_broadcasts"] == 3
        assert summary["total_responses"] == 5


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
        assert summary["deanon_success_rate"] == 0.25
        assert summary["eclipse_tokens_dropped"] == 7
        assert summary["correlation_evaluations"] == 2
        assert summary["correlation_success_rate"] == 0.5
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
        assert metrics.discrimination_score() == 1.0
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
        assert metrics.discrimination_score() == 1.0


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
