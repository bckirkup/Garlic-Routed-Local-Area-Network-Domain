"""Tests for continuous covariance forgetting while a device is worn."""

from __future__ import annotations

import numpy as np
import pytest

from garland.biometrics import BaselineTracker
from garland.config import config_from_dict, config_to_dict
from garland.hazards import PlumeConfig, SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig

LAMBDAS = [0.0, 0.001, 0.01, 0.1]


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


def _feed(tracker: BaselineTracker, scale: float, n: int, rng: np.random.Generator) -> None:
    width = len(tracker.channel_set)
    for step in range(n):
        tracker.update(tracker.mean_prior + rng.normal(0.0, scale, width), hour=step % 24, month=1)


def _contaminated_then_clean(lam: float, n_clean: int = 300) -> BaselineTracker:
    """Loud early residuals followed by quiet ones; returns the tracker."""
    rng = np.random.default_rng(3)
    tracker = BaselineTracker(covariance_forgetting_lambda=lam)
    _feed(tracker, scale=8.0, n=40, rng=rng)
    _feed(tracker, scale=1.0, n=n_clean, rng=rng)
    return tracker


class TestSensitivity:
    def test_higher_lambda_washes_out_early_contamination_faster(self):
        lambdas = [0.0, 0.001, 0.003, 0.01]
        traces = [np.trace(_contaminated_then_clean(lam).covariance_matrix()) for lam in lambdas]
        assert traces == sorted(traces, reverse=True), traces
        assert traces[-1] < 0.5 * traces[0], f"forgetting looks dead: {traces}"

    def test_counts_saturate_near_inverse_lambda(self):
        for lam in LAMBDAS[1:]:
            tracker = BaselineTracker(covariance_forgetting_lambda=lam)
            _feed(tracker, scale=1.0, n=int(20 / lam), rng=np.random.default_rng(1))
            expected = 1.0 / (1.0 - np.exp(-lam))
            np.testing.assert_allclose(tracker.cov_counts, expected, rtol=1e-6)

    def test_per_update_factor_is_exact(self):
        lam = 0.05
        tracker = BaselineTracker(covariance_forgetting_lambda=lam)
        rng = np.random.default_rng(5)
        _feed(tracker, scale=1.0, n=10, rng=rng)
        sums_before = tracker.cov_sum.copy()
        counts_before = tracker.cov_counts.copy()
        width = len(tracker.channel_set)
        observation = tracker.mean_prior + rng.normal(0.0, 1.0, width)
        residual = observation - tracker.expected_baseline(10, 1)
        tracker.update(observation, hour=10, month=1)
        np.testing.assert_allclose(
            tracker.cov_sum, sums_before * np.exp(-lam) + np.outer(residual, residual)
        )
        np.testing.assert_allclose(tracker.cov_counts, counts_before * np.exp(-lam) + 1.0)


class TestInvariants:
    def test_zero_lambda_reproduces_unbounded_running_sums(self):
        legacy = BaselineTracker()
        zero = BaselineTracker(covariance_forgetting_lambda=0.0)
        _feed(legacy, 1.0, 60, np.random.default_rng(9))
        _feed(zero, 1.0, 60, np.random.default_rng(9))
        np.testing.assert_array_equal(zero.cov_sum, legacy.cov_sum)
        np.testing.assert_array_equal(zero.cov_counts, legacy.cov_counts)
        assert float(zero.cov_counts[0, 0]) == pytest.approx(60.0, abs=0.0)

    def test_forgotten_covariance_stays_finite_and_positive_definite(self):
        tracker = _contaminated_then_clean(0.1, n_clean=2000)
        cov = tracker.covariance_matrix()
        assert np.all(np.isfinite(cov))
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)

    def test_forgetting_leaves_mean_and_sample_count_alone(self):
        with_forgetting = BaselineTracker(covariance_forgetting_lambda=0.1)
        without = BaselineTracker()
        _feed(with_forgetting, 1.0, 50, np.random.default_rng(2))
        _feed(without, 1.0, 50, np.random.default_rng(2))
        np.testing.assert_array_equal(with_forgetting.ema, without.ema)
        assert with_forgetting.n_samples == without.n_samples == 50

    def test_masked_channels_are_not_re_forgotten_into_zero(self):
        tracker = BaselineTracker(covariance_forgetting_lambda=0.1)
        width = len(tracker.channel_set)
        mask = np.ones(width, dtype=np.bool_)
        mask[0] = False
        rng = np.random.default_rng(4)
        for step in range(30):
            tracker.update(
                tracker.mean_prior + rng.normal(0.0, 1.0, width), step % 24, 1, observed=mask
            )
        assert tracker.cov_counts[0, 0] == pytest.approx(0.0, abs=0.0)
        assert tracker.cov_counts[1, 1] > 0.0
        assert np.all(np.isfinite(tracker.covariance_matrix()))

    @pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
    def test_rejects_invalid_lambda(self, bad):
        with pytest.raises(ValueError, match="covariance_forgetting_lambda"):
            BaselineTracker(covariance_forgetting_lambda=bad)


class TestModelIntegration:
    def test_config_round_trip_preserves_lambda(self):
        restored = config_from_dict(
            config_to_dict(_config(baseline_covariance_forgetting_lambda=0.02))
        )
        assert restored.baseline_covariance_forgetting_lambda == pytest.approx(0.02, abs=0.0)

    def test_model_plumbs_lambda_into_every_tracker(self):
        model = GarlandModel(_config(baseline_covariance_forgetting_lambda=0.02))
        assert all(
            b.covariance_forgetting_lambda == pytest.approx(0.02, abs=0.0) for b in model.baselines
        )

    def test_lambda_changes_run_but_zero_is_bit_identical_to_default(self):
        reference = GarlandModel(_config())
        reference.run(steps=12)
        zero = GarlandModel(_config(baseline_covariance_forgetting_lambda=0.0))
        zero.run(steps=12)
        for a, b in zip(reference.citizen_agents, zero.citizen_agents):
            np.testing.assert_array_equal(a.baseline.cov_sum, b.baseline.cov_sum)
        assert reference.metrics.step_records == zero.metrics.step_records
        assert reference.rng.bit_generator.state == zero.rng.bit_generator.state

        forgetting = GarlandModel(_config(baseline_covariance_forgetting_lambda=0.1))
        forgetting.run(steps=12)
        counts_ref = max(float(a.baseline.cov_counts.max()) for a in reference.citizen_agents)
        counts_forget = max(float(a.baseline.cov_counts.max()) for a in forgetting.citizen_agents)
        assert counts_forget < counts_ref
