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

from garland.privacy import AnomalyType


@dataclass
class DetectionEvent:
    """Record of a hazard detection by the system."""

    step: int
    hazard_type: str  # "disease" or "toxin"
    anomaly_type: AnomalyType
    zone_id: int
    true_positive: bool
    agents_affected: int


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

    # Privacy tracking
    epsilon_per_step: list[float] = field(default_factory=list)
    total_queries_issued: int = 0
    total_responses: int = 0

    # Attack metrics
    sybil_false_alerts: int = 0
    deanon_attempts: int = 0
    deanon_successes: int = 0

    def record_disease_onset(self, step: int) -> None:
        """Mark the step when first infectious case appears."""
        if self.disease_onset_step is None:
            self.disease_onset_step = step

    def record_toxin_onset(self, step: int) -> None:
        """Mark the step when plume begins."""
        if self.toxin_onset_step is None:
            self.toxin_onset_step = step

    def record_detection(self, event: DetectionEvent) -> None:
        """Record a system detection event and update confusion matrix."""
        self.detection_events.append(event)

        if event.hazard_type == "disease":
            if event.true_positive:
                self.true_positives_disease += 1
                if self.disease_detection_step is None:
                    self.disease_detection_step = event.step
            else:
                self.false_positives_disease += 1
        elif event.hazard_type == "toxin":
            if event.true_positive:
                self.true_positives_toxin += 1
                if self.toxin_detection_step is None:
                    self.toxin_detection_step = event.step
            else:
                self.false_positives_toxin += 1

    def record_missed_detection(self, hazard_type: str) -> None:
        """Record a step where a hazard was active but not detected."""
        if hazard_type == "disease":
            self.false_negatives_disease += 1
        elif hazard_type == "toxin":
            self.false_negatives_toxin += 1

    def record_no_hazard_no_detection(self, hazard_type: str) -> None:
        """Record a step with no hazard and no detection (true negative)."""
        if hazard_type == "disease":
            self.true_negatives_disease += 1
        elif hazard_type == "toxin":
            self.true_negatives_toxin += 1

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
            }
        )
        self.epsilon_per_step.append(cumulative_epsilon)

    def time_to_detection_disease(self) -> float | None:
        """Time (in 5-min steps) from disease onset to detection."""
        if self.disease_onset_step is None or self.disease_detection_step is None:
            return None
        return float(self.disease_detection_step - self.disease_onset_step)

    def time_to_detection_toxin(self) -> float | None:
        """Time (in 5-min steps) from toxin onset to detection."""
        if self.toxin_onset_step is None or self.toxin_detection_step is None:
            return None
        return float(self.toxin_detection_step - self.toxin_onset_step)

    def false_positive_rate_disease(self) -> float:
        """FPR = FP / (FP + TN) for disease detection."""
        denom = self.false_positives_disease + self.true_negatives_disease
        return self.false_positives_disease / denom if denom > 0 else 0.0

    def false_negative_rate_disease(self) -> float:
        """FNR = FN / (FN + TP) for disease detection."""
        denom = self.false_negatives_disease + self.true_positives_disease
        return self.false_negatives_disease / denom if denom > 0 else 0.0

    def false_positive_rate_toxin(self) -> float:
        """FPR = FP / (FP + TN) for toxin detection."""
        denom = self.false_positives_toxin + self.true_negatives_toxin
        return self.false_positives_toxin / denom if denom > 0 else 0.0

    def false_negative_rate_toxin(self) -> float:
        """FNR = FN / (FN + TP) for toxin detection."""
        denom = self.false_negatives_toxin + self.true_positives_toxin
        return self.false_negatives_toxin / denom if denom > 0 else 0.0

    def discrimination_score(self) -> float:
        """Measures system's ability to decouple toxin from disease.

        Score in [0, 1] where 1 = perfect discrimination.
        Based on correct classification of anomaly types.
        """
        correct = 0
        total = 0
        for event in self.detection_events:
            total += 1
            if event.hazard_type == "disease" and event.anomaly_type in (
                AnomalyType.FEBRILE,
                AnomalyType.MULTI_SYSTEM,
            ):
                correct += 1
            elif event.hazard_type == "toxin" and event.anomaly_type == AnomalyType.RESPIRATORY:
                correct += 1
        return correct / total if total > 0 else 0.0

    def summary(self) -> dict:
        """Generate summary metrics dictionary."""
        return {
            "time_to_detection_disease_steps": self.time_to_detection_disease(),
            "time_to_detection_disease_hours": (
                self.time_to_detection_disease() * 5 / 60
                if self.time_to_detection_disease() is not None
                else None
            ),
            "time_to_detection_toxin_steps": self.time_to_detection_toxin(),
            "time_to_detection_toxin_hours": (
                self.time_to_detection_toxin() * 5 / 60
                if self.time_to_detection_toxin() is not None
                else None
            ),
            "fpr_disease": self.false_positive_rate_disease(),
            "fnr_disease": self.false_negative_rate_disease(),
            "fpr_toxin": self.false_positive_rate_toxin(),
            "fnr_toxin": self.false_negative_rate_toxin(),
            "discrimination_score": self.discrimination_score(),
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
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert step records to pandas DataFrame."""
        return pd.DataFrame(self.step_records)

    def export_csv(self, path: str | Path) -> None:
        """Export step-level metrics to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def plot_metrics(self, output_dir: str | Path) -> None:
        """Generate diagnostic plots for simulation evaluation.

        Creates:
        1. SEIR curve (S, E, I, R over time)
        2. Detection timeline (hazard onset vs. system detection)
        3. Epsilon budget over time
        4. FP/FN rates
        """
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()

        if df.empty:
            return

        # 1. SEIR Curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["time_hours"], df["susceptible"], label="Susceptible", alpha=0.7)
        ax.plot(df["time_hours"], df["exposed"], label="Exposed", alpha=0.7)
        ax.plot(df["time_hours"], df["infectious"], label="Infectious", alpha=0.7)
        ax.plot(df["time_hours"], df["recovered"], label="Recovered", alpha=0.7)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Agent Count")
        ax.set_title("SEIR Dynamics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "seir_curve.png", dpi=150, bbox_inches="tight")
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
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Count")
        ax.set_title("Hazard Detection Timeline")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "detection_timeline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 3. Epsilon Budget
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["time_hours"], df["cumulative_epsilon"], color="purple")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Cumulative ε")
        ax.set_title("Privacy Budget Expenditure (Adaptive Composition)")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "epsilon_budget.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 4. System activity
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["time_hours"], df["tokens_submitted"], label="Tokens Submitted")
        ax.plot(df["time_hours"], df["broadcasts_issued"], label="Broadcasts Issued")
        ax.plot(df["time_hours"], df["responses_received"], label="Responses")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Count")
        ax.set_title("Privacy Protocol Activity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "protocol_activity.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
