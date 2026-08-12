"""Scheduled long-run sentinel for null-baseline stationarity."""

from __future__ import annotations

import pytest

from garland.hazards import SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig


@pytest.mark.slow
def test_null_baseline_false_anomaly_rate_stays_stationary():
    model = GarlandModel(
        SimulationConfig(
            n_agents=100,
            wearable_fraction=0.15,
            n_steps=8640,
            seed=42,
            mobility_model="static",
            grid_width=2000.0,
            grid_height=2000.0,
            cell_size=200.0,
            seir=SEIRConfig(initial_infected=0, outbreaks=[]),
            plumes=[],
        )
    )
    model.run()

    rates = []
    for day in range(1, 30):
        rows = model.metrics.step_records[day * 288 : (day + 1) * 288]
        active = sum(int(row["wearables_active"]) for row in rows)
        anomalies = sum(int(row["anomalies_detected"]) for row in rows)
        rates.append(anomalies / active)

    # The fixed implementation stays near 0.33–0.46% per operational wearable
    # step. The pre-#71 defect rose roughly tenfold across the month, so this
    # broad upper bound leaves margin for seed noise while decisively rejecting
    # that divergent trajectory.
    assert max(rates) < 0.02
    assert max(rates) - min(rates) < 0.015
