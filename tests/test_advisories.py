"""Tests for device-local advisory assembly and public confirmation tiers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from garland.advisories import AdvisoryConfig, AdvisoryEngine
from garland.privacy import AggregatorState, AnomalyType, BroadcastQuery


def _agent(idx: int, *, anomaly: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        idx=idx,
        anomaly_active=anomaly,
        anomaly_type=AnomalyType.FEBRILE,
        anomaly_onset_step=7,
        advisory=None,
        has_wearable=True,
        is_operational=True,
    )


def _query() -> BroadcastQuery:
    return BroadcastQuery(
        zone_cells=[4, 9],
        anomaly_type=AnomalyType.FEBRILE,
        time_window_start=0,
        time_window_end=12,
        query_id=22,
    )


def test_matching_local_anomaly_uses_device_onset_and_expires():
    agent = _agent(1)
    engine = AdvisoryEngine(AdvisoryConfig(advisory_expiry_steps=3), np.random.default_rng(1))

    issued = engine.refresh([_query()], {4: [agent], 9: []}, 8)

    assert len(issued) == 1
    assert agent.advisory.estimated_exposure_step == 7
    assert agent.advisory.tier == 1

    engine.refresh([], {4: [agent], 9: []}, 11)
    assert agent.advisory is None


def test_published_confirmation_upgrades_non_contributor():
    contributor = _agent(1)
    non_contributor = _agent(2)
    engine = AdvisoryEngine(
        AdvisoryConfig(
            clinic_visit_rate_per_day=288.0,
            advisory_confirmation_epsilon=1000.0,
            tier2_confirmations=1,
            tier3_confirmations=2,
        ),
        np.random.default_rng(2),
    )
    cells = {4: [contributor, non_contributor], 9: []}
    engine.refresh([_query()], cells, 0)

    result = engine.process_step(
        [contributor, non_contributor],
        1,
        disease_exposed={contributor.idx},
        toxin_exposed=set(),
    )

    assert result.released_counts
    assert non_contributor.advisory.tier >= 2


def test_disabled_advisory_metrics_are_absent_and_epsilon_is_zero():
    from garland.metrics import MetricsCollector

    metrics = MetricsCollector()
    metrics.configure_advisory_accounting(False)
    summary = metrics.summary()

    assert "advisories" not in summary
    assert metrics.advisory_confirmation_epsilon == pytest.approx(0.0)


def test_confirmation_release_composes_into_aggregator_epsilon():
    state = AggregatorState()

    state.record_advisory_confirmation_release(0.05)
    state.record_advisory_confirmation_release(0.05)

    assert state.advisory_confirmation_release_count == 2
    assert state.advisory_confirmation_epsilon == pytest.approx(0.1)
    assert state.total_epsilon >= state.advisory_confirmation_epsilon
