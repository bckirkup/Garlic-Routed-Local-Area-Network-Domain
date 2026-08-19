"""Regression and sensitivity tests for device-local baseline maturation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from garland.adoption import AdoptionConfig
from garland.baseline_maturation import BaselineMaturationConfig
from garland.constants import STEPS_PER_DAY
from garland.hazards import PlumeConfig, SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig


def _config(**kwargs) -> SimulationConfig:
    values = {
        "n_agents": 80,
        "wearable_fraction": 0.5,
        "grid_width": 1500.0,
        "grid_height": 1500.0,
        "n_steps": 4,
        "seed": 17,
        "world_settling_steps": 0,
        "seir": SEIRConfig(initial_infected=0),
        "plumes": [PlumeConfig(start_step=10_000)],
    }
    values.update(kwargs)
    return SimulationConfig(**values)


def _matured_model(days: int, cadence_steps: int = 12) -> GarlandModel:
    return GarlandModel(
        _config(
            baseline_maturation=BaselineMaturationConfig(
                minimum_history_days=days,
                maximum_history_days=days,
                cadence_steps=cadence_steps,
            )
        )
    )


def test_month_advances_and_matches_day_of_year():
    instance = GarlandModel(
        _config(n_steps=STEPS_PER_DAY * 45, start_datetime=datetime(2024, 1, 15))
    )
    observed_months = []
    for step in (0, STEPS_PER_DAY * 20, STEPS_PER_DAY * 45):
        instance.current_step = step
        _, _, month, day_of_year = instance._current_time_info()
        expected_month = next(
            index + 1
            for index, end in enumerate((31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365))
            if day_of_year <= end
        )
        observed_months.append(month)
        assert month == expected_month
    assert len(set(observed_months)) > 1


def test_history_length_increases_samples_and_month_coverage():
    models = [_matured_model(days) for days in (0, 7, 90, 365)]
    samples = [model.citizen_agents[0].baseline.n_samples for model in models]
    monthly = [
        int(np.count_nonzero(np.any(model.citizen_agents[0].baseline.monthly_counts > 1, axis=1)))
        for model in models
    ]
    assert samples == sorted(samples)
    assert samples[-1] > samples[1] > samples[0]
    assert monthly == sorted(monthly)
    assert monthly[-1] > monthly[1] >= monthly[0]


def test_samples_match_history_and_heterogeneous_draws_stay_in_bounds():
    model = GarlandModel(
        _config(
            baseline_maturation=BaselineMaturationConfig(
                minimum_history_days=7,
                maximum_history_days=9,
                cadence_steps=12,
            )
        )
    )
    fleet_start = [agent for agent in model.citizen_agents if agent.fleet_start_adopter]
    histories = model.baseline_maturation_history_days[model.baseline_maturation_history_days > 0]
    assert len(histories) == len(fleet_start)
    assert np.all((histories >= 7) & (histories <= 9))
    for agent, history_days in zip(fleet_start, histories, strict=True):
        assert agent.baseline.n_samples == int(history_days) * STEPS_PER_DAY // 12


def test_cadence_reduces_samples_but_preserves_monthly_coverage():
    fine = _matured_model(90, cadence_steps=12)
    coarse = _matured_model(90, cadence_steps=48)
    assert coarse.citizen_agents[0].baseline.n_samples < fine.citizen_agents[0].baseline.n_samples
    fine_months = np.count_nonzero(
        np.any(fine.citizen_agents[0].baseline.monthly_counts > 1, axis=1)
    )
    coarse_months = np.count_nonzero(
        np.any(coarse.citizen_agents[0].baseline.monthly_counts > 1, axis=1)
    )
    assert coarse_months == fine_months


def test_matured_baselines_are_finite_and_in_physiological_ranges():
    model = _matured_model(90, cadence_steps=12)
    for agent in model.citizen_agents:
        assert np.all(np.isfinite(agent.baseline.ema))
        # Resting ranges describe quiescent values, not observations: activity,
        # circadian variation, and measurement noise must remain learnable.
        excursions = np.array(
            [
                abs(channel.circadian_amp_max * channel.circadian_scale)
                + abs(channel.seasonal_coefficient)
                + abs(channel.activity_coefficient) * 0.35
                + 4.0 * channel.noise_sd
                for channel in model.channel_set.channels
            ]
        )
        resting_lower = np.array([channel.resting_min for channel in model.channel_set.channels])
        resting_upper = np.array([channel.resting_max for channel in model.channel_set.channels])
        plausible_lower = resting_lower - excursions
        plausible_upper = resting_upper + excursions
        assert np.all(agent.baseline.ema >= model.channel_set.hard_floors)
        assert np.all(agent.baseline.ema >= plausible_lower)
        assert np.all(agent.baseline.ema <= plausible_upper)


def test_maturation_is_device_local_and_does_not_spend_privacy():
    model = _matured_model(7, cadence_steps=12)
    assert model.aggregator.state.total_epsilon == pytest.approx(0.0)
    assert model.metrics.detection_events == []


def test_cold_baseline_rate_falls_with_history():
    fractions = []
    for days in (0, 7, 90):
        model = _matured_model(days)
        model.step()
        fractions.append(model.metrics.summary()["fleet_cold_baseline_wearable_step_fraction"])
    assert fractions == sorted(fractions, reverse=True)
    assert fractions[0] > fractions[-1]


def test_maturation_does_not_raise_midday_benign_anomaly_rate():
    configs = [
        BaselineMaturationConfig(),
        BaselineMaturationConfig(
            minimum_history_days=90,
            maximum_history_days=90,
            cadence_steps=12,
        ),
    ]
    rates = []
    for maturation in configs:
        model = GarlandModel(
            _config(
                n_agents=80,
                wearable_fraction=1.0,
                mobility_model="static",
                n_steps=STEPS_PER_DAY,
                start_datetime=datetime(2024, 1, 15),
                baseline_maturation=maturation,
            )
        )
        for _ in range(STEPS_PER_DAY):
            model.step()
        midday = [
            record for record in model.metrics.step_records if 10 <= record["time_hours"] < 16
        ]
        rates.append(
            sum(record["anomalies_detected"] for record in midday)
            / sum(record["wearables_active"] for record in midday)
        )

    cold_rate, matured_rate = rates
    assert matured_rate <= cold_rate + 0.01
    assert matured_rate < cold_rate


def test_disabled_maturation_preserves_main_loop_random_draws():
    baseline = _config()
    disabled = GarlandModel(baseline)
    explicit = GarlandModel(
        replace(
            baseline,
            baseline_maturation=BaselineMaturationConfig(),
        )
    )
    for _ in range(4):
        disabled.step()
        explicit.step()
    np.testing.assert_array_equal(disabled.agent_x, explicit.agent_x)
    np.testing.assert_array_equal(disabled.agent_y, explicit.agent_y)
    assert disabled.metrics.summary() == explicit.metrics.summary()


def test_maturation_rng_does_not_change_main_loop_draws():
    no_history = GarlandModel(_config())
    with_history = GarlandModel(
        _config(
            baseline_maturation=BaselineMaturationConfig(
                minimum_history_days=7, maximum_history_days=7
            )
        )
    )
    np.testing.assert_array_equal(
        no_history.rng.integers(0, 2**31, size=16),
        with_history.rng.integers(0, 2**31, size=16),
    )


def test_mid_run_adopters_receive_no_prior_history():
    model = GarlandModel(
        _config(
            adoption=AdoptionConfig(
                mode="rollout",
                start_step=1,
                initial_adopted_fraction=0.0,
                rate=1.0,
            ),
            baseline_maturation=BaselineMaturationConfig(
                minimum_history_days=7, maximum_history_days=7
            ),
        )
    )
    assert all(not agent.fleet_start_adopter for agent in model.citizen_agents)
    assert all(agent.baseline.n_samples == 0 for agent in model.citizen_agents)
    model.step()
    model.step()
    operational = [agent for agent in model.citizen_agents if agent.is_operational]
    assert operational
    assert all(agent.baseline.n_samples > 0 for agent in operational)
