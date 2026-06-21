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
        metrics = MetricsCollector()
        metrics.sync_attack_metrics(deanon_attempts=4, deanon_successes=1)
        summary = metrics.summary()
        assert metrics.deanon_attempts == 4
        assert metrics.deanon_successes == 1
        assert summary["deanon_success_rate"] == 0.25
