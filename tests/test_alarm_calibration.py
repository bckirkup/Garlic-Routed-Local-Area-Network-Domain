"""Tests for the fleet alarm-rate calibration that flattens FPR against width."""

from __future__ import annotations

import numpy as np
import pytest

from garland.alarm_calibration import (
    RATIO_BIN_WIDTH,
    RATIO_BINS,
    AlarmCalibrationConfig,
    AlarmRateCalibrator,
    calibrator_for,
)
from garland.detection_power import WIDTH_BUCKET_LABELS
from garland.hazards import SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig
from garland.thresholds import REFERENCE_DOF, per_epoch_false_positive_rate, threshold_for_dof


def _feed(
    calibrator: AlarmRateCalibrator,
    width: int,
    ratios: np.ndarray,
) -> None:
    """Open the window and fold a whole sample of ratios in at one width."""
    calibrator.advance(calibrator.config.start_step)
    for ratio in ratios:
        calibrator.observe(width, float(ratio))


def _calibrator(**overrides: object) -> AlarmRateCalibrator:
    config = AlarmCalibrationConfig(**overrides)  # type: ignore[arg-type]
    return calibrator_for(config, 3.5)


class TestConfigValidation:
    """The window and bounds have to be coherent before a run starts."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"start_step": -1},
            {"end_step": 0},
            {"start_step": 500, "end_step": 500},
            {"max_scale": 0.99},
            {"min_samples": 0},
        ],
    )
    def test_incoherent_settings_are_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            AlarmCalibrationConfig(**overrides)  # type: ignore[arg-type]

    def test_defaults_are_accepted_and_calibrate_before_a_default_run_ends(self) -> None:
        config = AlarmCalibrationConfig()
        assert config.start_step < config.end_step
        assert config.max_scale >= 1.0

    def test_target_rate_must_be_a_probability(self) -> None:
        with pytest.raises(ValueError):
            AlarmRateCalibrator(target_rate=0.0)
        with pytest.raises(ValueError):
            AlarmRateCalibrator(target_rate=1.0)


class TestWindow:
    """Collection is confined to the configured window and then frozen."""

    def test_window_opens_holds_and_closes(self) -> None:
        calibrator = _calibrator(start_step=10, end_step=20)
        calibrator.advance(0)
        assert not calibrator.collecting
        calibrator.observe(8, 2.0)
        assert calibrator.samples == 0

        calibrator.advance(10)
        assert calibrator.collecting
        calibrator.observe(8, 2.0)
        assert calibrator.samples == 1

        calibrator.advance(20)
        assert calibrator.frozen
        assert not calibrator.collecting
        calibrator.observe(8, 2.0)
        assert calibrator.samples == 1

    def test_freezing_is_idempotent_and_keeps_the_first_scales(self) -> None:
        calibrator = _calibrator(start_step=0, end_step=10, min_samples=10)
        _feed(calibrator, 8, np.linspace(0.1, 1.6, 2000))
        calibrator.freeze()
        learned = dict(calibrator.scales)
        _feed(calibrator, 8, np.full(2000, 5.0))
        calibrator.freeze()
        assert calibrator.scales == learned

    def test_non_finite_and_silent_epochs_are_ignored(self) -> None:
        calibrator = _calibrator(start_step=0, end_step=10)
        calibrator.advance(0)
        calibrator.observe(0, 1.0)
        calibrator.observe(8, float("nan"))
        calibrator.observe(8, float("inf"))
        assert calibrator.samples == 0


class TestScaleEstimation:
    """The learned scale tracks how heavy the observed null tail actually is."""

    @staticmethod
    def _empirical_rate(calibrator: AlarmRateCalibrator, width: int, ratios: np.ndarray) -> float:
        return float((ratios > calibrator.scale_for(width)).mean())

    @pytest.mark.parametrize("tail_scale", [1.0, 1.5, 2.0, 3.0])
    def test_heavier_tails_earn_larger_scales(self, tail_scale: float) -> None:
        """A few different tail weights produce a few different scales, in order."""
        rng = np.random.default_rng(5)
        calibrator = _calibrator(start_step=0, end_step=10, max_scale=10.0)
        ratios = np.abs(rng.normal(0.0, 0.3, 40_000)) * tail_scale
        _feed(calibrator, 8, ratios)
        calibrator.freeze()
        scale = calibrator.scale_for(8)
        assert scale >= 1.0
        # The cut it learned reproduces the target rate on the sample it saw.
        target = per_epoch_false_positive_rate(3.5, REFERENCE_DOF)
        rate = self._empirical_rate(calibrator, 8, ratios)
        assert rate <= target + 0.005

    def test_scale_is_monotone_in_tail_weight(self) -> None:
        rng = np.random.default_rng(6)
        scales = []
        for tail_scale in (1.0, 2.0, 4.0):
            calibrator = _calibrator(start_step=0, end_step=10, max_scale=20.0)
            _feed(calibrator, 8, np.abs(rng.normal(0.0, 0.3, 40_000)) * tail_scale)
            calibrator.freeze()
            scales.append(calibrator.scale_for(8))
        assert scales == sorted(scales)
        assert scales[-1] > scales[0]

    def test_a_null_that_already_matches_target_is_left_alone(self) -> None:
        """Ratios whose tail already sits at target must not lose sensitivity."""
        rng = np.random.default_rng(7)
        calibrator = _calibrator(start_step=0, end_step=10)
        target = per_epoch_false_positive_rate(3.5, REFERENCE_DOF)
        ratios = np.abs(rng.normal(0.0, 1.0, 60_000))
        # Rescale so exactly the target fraction sits above 1.0.
        ratios = ratios / np.quantile(ratios, 1.0 - target)
        _feed(calibrator, 8, ratios)
        calibrator.freeze()
        assert calibrator.scale_for(8) == pytest.approx(1.0, abs=0.05)

    def test_scale_is_clamped_between_one_and_max(self) -> None:
        calibrator = _calibrator(start_step=0, end_step=10, max_scale=1.5)
        _feed(calibrator, 8, np.full(5_000, 8.0))
        calibrator.freeze()
        assert calibrator.scale_for(8) == pytest.approx(1.5)

        quiet = _calibrator(start_step=0, end_step=10)
        _feed(quiet, 8, np.full(5_000, 0.01))
        quiet.freeze()
        assert quiet.scale_for(8) == pytest.approx(1.0)

    def test_unresolvable_tail_leaves_the_bucket_uncalibrated(self) -> None:
        """Every ratio above the histogram range gives no quantile to read."""
        calibrator = _calibrator(start_step=0, end_step=10)
        overflow = (RATIO_BINS + 5) * RATIO_BIN_WIDTH
        _feed(calibrator, 8, np.full(5_000, overflow))
        calibrator.freeze()
        assert calibrator.scale_for(8) == pytest.approx(1.0)

    def test_empty_window_leaves_every_scale_neutral(self) -> None:
        calibrator = _calibrator(start_step=0, end_step=10)
        calibrator.freeze()
        assert set(calibrator.scales) == set(WIDTH_BUCKET_LABELS)
        assert all(scale == pytest.approx(1.0) for scale in calibrator.scales.values())

    def test_thin_bucket_borrows_its_nearest_calibrated_neighbour(self) -> None:
        """A bucket below ``min_samples`` inherits rather than fitting noise."""
        rng = np.random.default_rng(8)
        calibrator = _calibrator(start_step=0, end_step=10, min_samples=1_000, max_scale=10.0)
        _feed(calibrator, 8, np.abs(rng.normal(0.0, 0.6, 40_000)))
        _feed(calibrator, 30, np.array([0.01, 0.02, 0.03]))
        calibrator.freeze()
        assert calibrator.scale_for(30) == pytest.approx(calibrator.scale_for(8))
        assert calibrator.scale_for(30) > 1.0

    def test_the_borrowed_neighbour_is_the_closest_width_not_the_fleet_pool(self) -> None:
        """Narrow epochs dominate the pool, so the pool would under-correct."""
        rng = np.random.default_rng(11)
        calibrator = _calibrator(start_step=0, end_step=10, min_samples=1_000, max_scale=10.0)
        # A large near-target narrow population and a smaller heavy-tailed
        # mid-width one; the thin widest bucket must follow the mid width.
        _feed(calibrator, 3, np.abs(rng.normal(0.0, 0.25, 200_000)))
        _feed(calibrator, 8, np.abs(rng.normal(0.0, 0.9, 20_000)))
        _feed(calibrator, 30, np.array([0.01, 0.02]))
        calibrator.freeze()
        assert calibrator.scale_for(8) > calibrator.scale_for(3)
        assert calibrator.scale_for(30) == pytest.approx(calibrator.scale_for(8))

    def test_a_thick_bucket_fits_its_own_tail(self) -> None:
        rng = np.random.default_rng(9)
        calibrator = _calibrator(start_step=0, end_step=10, min_samples=1_000, max_scale=10.0)
        _feed(calibrator, 8, np.abs(rng.normal(0.0, 0.3, 40_000)))
        _feed(calibrator, 20, np.abs(rng.normal(0.0, 0.9, 40_000)))
        calibrator.freeze()
        assert calibrator.scale_for(20) > calibrator.scale_for(8)

    def test_repeated_calibration_is_deterministic(self) -> None:
        runs = []
        for _ in range(2):
            calibrator = _calibrator(start_step=0, end_step=10, max_scale=10.0)
            _feed(calibrator, 8, np.abs(np.random.default_rng(3).normal(0.0, 0.5, 20_000)))
            calibrator.freeze()
            runs.append(dict(calibrator.scales))
        assert runs[0] == runs[1]


class TestThresholdApplication:
    """The scale multiplies the width-aware cut and nothing else."""

    def test_threshold_scales_the_dof_calibrated_cut(self) -> None:
        calibrator = _calibrator(start_step=0, end_step=10, max_scale=10.0)
        _feed(calibrator, 8, np.abs(np.random.default_rng(4).normal(0.0, 0.6, 40_000)))
        calibrator.freeze()
        for width in (2, 8, 16, 28):
            expected = threshold_for_dof(3.5, width) * calibrator.scale_for(width)
            assert calibrator.threshold(3.5, width) == pytest.approx(expected)

    def test_scale_is_looked_up_per_width_bucket(self) -> None:
        calibrator = _calibrator(start_step=0, end_step=10)
        calibrator.scales.update({"1-5": 1.1, "6-12": 1.2, "13-24": 1.3, "25+": 1.4})
        assert calibrator.scale_for(3) == pytest.approx(1.1)
        assert calibrator.scale_for(12) == pytest.approx(1.2)
        assert calibrator.scale_for(24) == pytest.approx(1.3)
        assert calibrator.scale_for(40) == pytest.approx(1.4)

    def test_target_rate_follows_the_configured_threshold(self) -> None:
        lenient = calibrator_for(AlarmCalibrationConfig(), 3.0)
        strict = calibrator_for(AlarmCalibrationConfig(), 4.5)
        assert lenient.target_rate > strict.target_rate

    def test_summary_reports_the_evidence_behind_the_scales(self) -> None:
        calibrator = _calibrator(start_step=0, end_step=10)
        _feed(calibrator, 8, np.abs(np.random.default_rng(2).normal(0.0, 0.5, 3_000)))
        assert calibrator.summary()["frozen"] is False
        calibrator.freeze()
        summary = calibrator.summary()
        assert summary["frozen"] is True
        assert summary["calibration_epochs"] == 3_000
        assert set(summary["scales"]) == set(WIDTH_BUCKET_LABELS)  # type: ignore[arg-type]


def _town(**overrides: object) -> GarlandModel:
    """Quiet mixed-width town: no hazard, no confounder, every subsystem worn."""
    config = SimulationConfig(
        n_agents=300,
        wearable_fraction=0.5,
        n_steps=360,
        seed=17,
        mobility_model="static",
        world_settling_steps=0,
        seir=SEIRConfig(beta=0.0, initial_infected=0),
        plumes=[],
    )
    config.confounders.enabled = False
    config.devices.enabled = True
    config.devices.adoption = {
        "motion_actigraphy": 1.0,
        "instrumented_footwear": 1.0,
        "respiratory_acoustic_patch": 1.0,
        "chest_electrode_patch": 1.0,
    }
    config.alarm_calibration = AlarmCalibrationConfig(
        start_step=60,
        end_step=180,
        **overrides,  # type: ignore[arg-type]
    )
    model = GarlandModel(config)
    model.run()
    return model


class TestSimulationWiring:
    """One shared calibrator, frozen mid-run, visible in detection power."""

    def test_the_fleet_shares_one_calibrator(self) -> None:
        model = _town()
        calibrators = {id(agent.alarm_calibrator) for agent in model.citizen_agents}
        assert calibrators == {id(model.alarm_calibrator)}

    def test_calibration_freezes_and_is_reported(self) -> None:
        model = _town()
        summary = model.metrics.summary()["detection_power"]["alarm_calibration"]
        assert summary["frozen"] is True
        assert summary["calibration_epochs"] > 0
        assert all(1.0 <= scale <= 3.0 for scale in summary["scales"].values())

    def test_disabling_calibration_removes_it_entirely(self) -> None:
        config = _town().config
        config.alarm_calibration = AlarmCalibrationConfig(enabled=False)
        model = GarlandModel(config)
        model.run()
        assert model.alarm_calibrator is None
        assert all(agent.alarm_calibrator is None for agent in model.citizen_agents)
        assert "alarm_calibration" not in model.metrics.summary()["detection_power"]

    def test_sequential_mode_is_not_calibrated(self) -> None:
        """CUSUM accumulates its own evidence; the instant cut is not its cut."""
        config = _town().config
        config.detector_mode = "sequential"
        model = GarlandModel(config)
        assert model.alarm_calibrator is None

    def test_calibration_raises_the_cut_without_silencing_the_fleet(self) -> None:
        model = _town()
        buckets = model.metrics.summary()["detection_power"]["width_buckets"]
        assert any(bucket["tokens"] > 0 for bucket in buckets.values())
        wide = model.alarm_calibrator.scale_for(16)  # type: ignore[union-attr]
        assert wide > 1.0


FREEZE_STEP = 180


def _split_run(*, defer: bool, threshold_m: int = 2, n_steps: int = 360) -> tuple[int, int, float]:
    """Broadcasts before the freeze, after it, and the epsilon they cost.

    The scenario is quiet and hazard-free, so every broadcast here is a false
    one; pre-freeze broadcasts are the ones the gate is meant to withhold.
    """
    config = SimulationConfig(
        n_agents=300,
        wearable_fraction=0.5,
        n_steps=n_steps,
        seed=17,
        mobility_model="static",
        world_settling_steps=0,
        seir=SEIRConfig(beta=0.0, initial_infected=0),
        plumes=[],
    )
    config.confounders.enabled = False
    config.devices.enabled = True
    config.privacy.threshold_m = threshold_m
    config.privacy.k_min = 1
    config.alarm_calibration = AlarmCalibrationConfig(
        start_step=60,
        end_step=FREEZE_STEP,
        defer_broadcasts_until_frozen=defer,
    )
    model = GarlandModel(config)
    split = min(FREEZE_STEP, config.n_steps)
    for _ in range(split):
        model.step()
    before = model.metrics.total_queries_issued
    for _ in range(split, config.n_steps):
        model.step()
    after = model.metrics.total_queries_issued - before
    epsilon = float(model.metrics.summary()["total_epsilon"])
    return before, after, epsilon


class TestPreCalibrationBroadcastGate:
    """Zones stay quiet while the cut they would trigger on is still unknown."""

    def test_ungated_runs_broadcast_on_uncalibrated_tokens(self) -> None:
        """Negative control: without the gate the pre-freeze period broadcasts."""
        before, after, _ = _split_run(defer=False)
        assert before > 0
        assert after > 0

    def test_the_gate_withholds_pre_freeze_broadcasts_only(self) -> None:
        gated_before, gated_after, _ = _split_run(defer=True)
        assert gated_before == 0
        assert gated_after > 0

    def test_the_gate_leaves_post_freeze_volume_comparable(self) -> None:
        """Withholding early tokens must not silence the fleet afterwards."""
        _, gated_after, _ = _split_run(defer=True)
        _, ungated_after, _ = _split_run(defer=False)
        assert gated_after >= 0.5 * ungated_after

    def test_the_gate_spends_strictly_less_epsilon(self) -> None:
        _, _, gated = _split_run(defer=True)
        _, _, ungated = _split_run(defer=False)
        assert 0.0 < gated < ungated

    @pytest.mark.parametrize("threshold_m", [2, 4, 8])
    def test_the_gate_holds_at_every_trigger_count(self, threshold_m: int) -> None:
        before, _, _ = _split_run(defer=True, threshold_m=threshold_m)
        assert before == 0

    def test_broadcast_volume_still_falls_as_the_trigger_count_rises(self) -> None:
        """The gate must not flatten the threshold's effect on volume."""
        counts = [_split_run(defer=True, threshold_m=m)[1] for m in (2, 4, 8)]
        assert counts[0] > counts[1] > counts[2]

    def test_the_gate_is_off_by_default_because_short_runs_never_freeze(self) -> None:
        """The trap the default avoids: no freeze in the run, so no broadcasts.

        A scenario whose hazards land inside the calibration window would lose
        them outright, which is why enabling this is the scenario's decision.
        """
        assert AlarmCalibrationConfig().defer_broadcasts_until_frozen is False
        gated, ungated = (
            _split_run(defer=defer, n_steps=FREEZE_STEP // 2)[0] for defer in (True, False)
        )
        assert gated == 0
        assert ungated > 0
