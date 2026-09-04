"""Tests for covariance aging when a removed wearable is re-worn."""

from __future__ import annotations

import numpy as np
import pytest

from garland.biometrics import COVARIANCE_WARMUP_SAMPLES, BaselineTracker
from garland.device_lifecycle import DeviceLifecycleEngine, DeviceStatus
from garland.hazards import PlumeConfig, SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig


def _config(**kwargs) -> SimulationConfig:
    defaults = {
        "n_agents": 600,
        "wearable_fraction": 0.2,
        "grid_width": 2000.0,
        "grid_height": 2000.0,
        "n_steps": 20,
        "seed": 61,
        "seir": SEIRConfig(initial_infected=5, beta=0.03),
        "plumes": [PlumeConfig(start_step=10_000)],
    }
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


def _learned_tracker(decay_lambda: float = 0.01, n: int = 40) -> BaselineTracker:
    rng = np.random.default_rng(7)
    tracker = BaselineTracker(decay_lambda=decay_lambda)
    width = len(tracker.channel_set)
    for step in range(n):
        observation = tracker.mean_prior + rng.normal(0.0, 1.0, width)
        tracker.update(observation, hour=step % 24, month=1)
    return tracker


def _prior_distance(tracker: BaselineTracker) -> float:
    return float(np.abs(tracker.covariance_matrix() - tracker.covariance_prior).max())


class TestSensitivity:
    def test_longer_gap_pulls_covariance_further_toward_prior(self):
        gaps = [0, 24, 288, 2016]
        distances = []
        for gap in gaps:
            tracker = _learned_tracker()
            tracker.decay_covariance(gap)
            distances.append(_prior_distance(tracker))
        assert distances == sorted(distances, reverse=True)
        assert distances[0] > 0.0
        assert distances[-1] < 0.05 * distances[0], f"decay looks dead: {distances}"

    def test_higher_lambda_forgets_faster_over_same_gap(self):
        lambdas = [0.001, 0.01, 0.1]
        distances = []
        for lam in lambdas:
            tracker = _learned_tracker(decay_lambda=lam)
            tracker.decay_covariance(100)
            distances.append(_prior_distance(tracker) / _prior_distance(_learned_tracker(lam)))
        assert distances == sorted(distances, reverse=True)
        assert distances[0] - distances[-1] > 0.5

    def test_gap_decay_matches_mean_forgetting_factor(self):
        tracker = _learned_tracker(decay_lambda=0.02)
        sums_before = tracker.cov_sum.copy()
        counts_before = tracker.cov_counts.copy()
        tracker.decay_covariance(50)
        factor = np.exp(-0.02 * 50)
        np.testing.assert_allclose(tracker.cov_sum, sums_before * factor)
        np.testing.assert_allclose(tracker.cov_counts, counts_before * factor)


class TestInvariants:
    def test_zero_or_negative_gap_is_a_no_op(self):
        tracker = _learned_tracker()
        before = tracker.cov_sum.copy()
        tracker.decay_covariance(0)
        tracker.decay_covariance(-3)
        np.testing.assert_array_equal(tracker.cov_sum, before)

    def test_decayed_covariance_stays_finite_and_positive_definite(self):
        tracker = _learned_tracker()
        tracker.decay_covariance(50_000)
        cov = tracker.covariance_matrix()
        assert np.all(np.isfinite(cov))
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)
        np.testing.assert_allclose(cov, tracker.covariance_prior, atol=1e-9)

    def test_decay_does_not_touch_mean_or_sample_count(self):
        tracker = _learned_tracker()
        ema_before = tracker.ema.copy()
        n_before = tracker.n_samples
        tracker.decay_covariance(500)
        np.testing.assert_array_equal(tracker.ema, ema_before)
        assert tracker.n_samples == n_before
        assert n_before >= COVARIANCE_WARMUP_SAMPLES


def _model_with_lifecycle(**kwargs) -> GarlandModel:
    model = GarlandModel(_config(**kwargs))
    model.device_lifecycle_engine = DeviceLifecycleEngine(
        len(model.citizen_agents), model.config.device_lifecycle, model.rng
    )
    return model


def _remove_then_rewear(model: GarlandModel, gap: int) -> None:
    agent = model.citizen_agents[0]
    engine = model.device_lifecycle_engine
    assert engine is not None
    agent.baseline = _learned_tracker(model.config.baseline_decay_lambda)
    model.current_step = 10
    engine.status[0] = DeviceStatus.NOT_WORN
    model._sync_citizen_device_state()
    assert agent.device_status == DeviceStatus.NOT_WORN
    assert agent.device_removed_step == 10
    model.current_step = 10 + gap
    engine.status[0] = DeviceStatus.ACTIVE
    model._sync_citizen_device_state()
    assert agent.device_status == DeviceStatus.ACTIVE
    assert agent.device_removed_step is None


class TestLifecycleIntegration:
    def test_default_keeps_covariance_undecayed_across_rewear(self):
        model = _model_with_lifecycle()
        _remove_then_rewear(model, gap=288)
        agent = model.citizen_agents[0]
        reference = _learned_tracker(model.config.baseline_decay_lambda)
        np.testing.assert_array_equal(agent.baseline.cov_sum, reference.cov_sum)
        np.testing.assert_array_equal(agent.baseline.cov_counts, reference.cov_counts)

    def test_opt_in_ages_covariance_by_gap_length(self):
        gaps = [1, 288, 2016]
        distances = []
        for gap in gaps:
            model = _model_with_lifecycle(rewear_covariance_decay=True)
            _remove_then_rewear(model, gap=gap)
            distances.append(_prior_distance(model.citizen_agents[0].baseline))
        undecayed = _prior_distance(_learned_tracker(0.01))
        assert distances == sorted(distances, reverse=True)
        assert distances[0] < undecayed
        assert distances[-1] < 0.1 * undecayed

    def test_opt_in_applies_exact_forgetting_factor(self):
        model = _model_with_lifecycle(rewear_covariance_decay=True, baseline_decay_lambda=0.05)
        _remove_then_rewear(model, gap=30)
        reference = _learned_tracker(0.05)
        factor = np.exp(-0.05 * 30)
        np.testing.assert_allclose(
            model.citizen_agents[0].baseline.cov_sum, reference.cov_sum * factor
        )

    def test_config_round_trip_preserves_flag(self):
        from garland.config import config_from_dict, config_to_dict

        for value in (False, True):
            restored = config_from_dict(config_to_dict(_config(rewear_covariance_decay=value)))
            assert restored.rewear_covariance_decay is value


class TestRegression:
    @pytest.mark.parametrize("flag", [False, True])
    def test_run_without_removal_is_bit_identical_to_default(self, flag):
        """Without device removal the flag must not perturb any seeded output."""
        baseline = GarlandModel(_config())
        baseline.run(steps=12)
        candidate = GarlandModel(_config(rewear_covariance_decay=flag))
        candidate.run(steps=12)
        for reference, other in zip(baseline.citizen_agents, candidate.citizen_agents):
            np.testing.assert_array_equal(reference.baseline.cov_sum, other.baseline.cov_sum)
            np.testing.assert_array_equal(reference.baseline.ema, other.baseline.ema)
        assert baseline.metrics.step_records == candidate.metrics.step_records
        assert baseline.rng.bit_generator.state == candidate.rng.bit_generator.state
