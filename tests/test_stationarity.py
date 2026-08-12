"""Scheduled long-run sentinel for null-baseline stationarity."""

from __future__ import annotations

import pytest

from garland.hazards import SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig


@pytest.mark.slow
def test_null_baseline_false_anomaly_rate_stays_stationary():
    model = GarlandModel(
        SimulationConfig(
            n_agents=1000,
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

    # With 1000 agents, the fixed implementation measured 0.297–0.522% per
    # operational wearable step across days 1–29, with a 0.225 percentage-point
    # spread. These bounds leave room above that observed sampling noise while
    # decisively rejecting the pre-#71 trajectory, which rose to about 8.9%.
    assert max(rates) < 0.01
    assert max(rates) - min(rates) < 0.005
