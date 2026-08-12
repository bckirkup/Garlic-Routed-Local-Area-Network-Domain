"""Tests for sequential per-person detection."""

from __future__ import annotations

import numpy as np

from garland.detection import SequentialDetector
from garland.privacy import AnomalyType, classify_anomaly


def test_cusum_accumulates_and_starts_one_alarm() -> None:
    detector = SequentialDetector(reference_value=2.0, threshold=5.0)
    residual = np.array([0.0, 0.0, 5.0, 0.0])

    assert not detector.update(4.0, residual)
    assert detector.statistic == 2.0
    assert detector.update(6.0, residual)
    assert detector.alarm_active
    assert not detector.update(6.0, residual)


def test_cusum_hysteresis_requires_consecutive_zero_steps() -> None:
    detector = SequentialDetector(
        reference_value=1.0,
        threshold=0.9,
        clear_steps=2,
    )
    residual = np.zeros(4)
    assert detector.update(3.0, residual)

    assert not detector.update(0.0, residual)
    assert detector.alarm_active
    assert not detector.update(0.0, residual)
    assert detector.alarm_active
    assert not detector.update(0.0, residual)
    assert not detector.alarm_active


def test_sequential_classification_uses_accumulated_residual() -> None:
    detector = SequentialDetector(residual_ewma_alpha=0.5)
    respiratory = np.array([11.0, -12.0, 12.0, 0.0])

    detector.update(3.0, respiratory)
    detector.update(3.0, respiratory)

    assert classify_anomaly(detector.residual_ewma, np.zeros(4)) == AnomalyType.RESPIRATORY
