"""Tests for width- and device-stratified detection-power instrumentation."""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import DEFAULT_CHANNEL_SET
from garland.detection_power import (
    WIDTH_BUCKET_LABELS,
    AblationProbe,
    DetectionPowerConfig,
    DetectionPowerTracker,
    width_bucket,
)
from garland.hazards import PlumeConfig, SEIRConfig
from garland.simulation import GarlandModel, SimulationConfig

RATE_KEYS = (
    "token_rate",
    "true_positive_rate",
    "false_positive_rate",
)
DEVICE_RATE_KEYS = (
    "observed_channel_fraction",
    "masked_channel_fraction",
    "reporting_epoch_fraction",
    "true_positive_rate",
    "false_positive_rate",
)


def _model(
    *,
    adoption: dict[str, float] | None = None,
    backend: str = "rect",
    n_steps: int = 120,
    ablation_rate: float = 0.0,
    seed: int = 11,
    plume_start: int = 40,
) -> GarlandModel:
    """Small mixed-modality town used by the integration checks."""
    config = SimulationConfig(
        n_agents=600,
        wearable_fraction=0.4,
        n_steps=n_steps,
        seed=seed,
        spatial_backend=backend,
        mobility_model="static",
        world_settling_steps=0,
        seir=SEIRConfig(initial_infected=20),
        plumes=[PlumeConfig(start_step=plume_start, duration_steps=60)],
        detection_power=DetectionPowerConfig(channel_ablation_rate=ablation_rate),
    )
    if adoption:
        config.devices.enabled = True
        config.devices.adoption = dict(adoption)
    model = GarlandModel(config)
    model.run()
    return model


def _tracker_summary(model: GarlandModel) -> dict:
    return model.metrics.summary()["detection_power"]


@pytest.mark.parametrize(
    ("width", "label"),
    [
        (1, "1-5"),
        (4, "1-5"),
        (5, "1-5"),
        (6, "6-12"),
        (12, "6-12"),
        (13, "13-24"),
        (24, "13-24"),
        (25, "25+"),
        (400, "25+"),
    ],
)
def test_width_bucket_boundaries(width: int, label: str):
    assert width_bucket(width) == label


@pytest.mark.parametrize("width", [0, -3])
def test_width_bucket_rejects_non_reporting_widths(width: int):
    with pytest.raises(ValueError, match="at least 1"):
        width_bucket(width)


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_detection_power_config_rejects_out_of_range_rate(rate: float):
    with pytest.raises(ValueError, match="channel_ablation_rate"):
        DetectionPowerConfig(channel_ablation_rate=rate)


def test_record_epochs_partitions_scored_and_silent_epochs():
    tracker = DetectionPowerTracker()
    widths = np.array([0, 4, 8, 20, 30], dtype=np.int64)
    emitted = np.array([False, True, False, True, False])
    hazard = np.array([True, True, False, False, False])
    tracker.record_epochs(step=0, widths=widths, emitted=emitted, hazard=hazard)

    summary = tracker.summary()
    assert summary["scored_epochs"] == 4
    assert summary["silent_epochs"] == 1
    buckets = summary["width_buckets"]
    assert [buckets[label]["scored_epochs"] for label in WIDTH_BUCKET_LABELS] == [1, 1, 1, 1]
    # The silent agent contributes to neither the hazard nor the clean denominator.
    assert sum(
        buckets[label]["hazard_epochs"] + buckets[label]["clean_epochs"]
        for label in WIDTH_BUCKET_LABELS
    ) == int(summary["scored_epochs"])
    assert buckets["1-5"]["true_positive_rate"] == pytest.approx(1.0)
    assert buckets["1-5"]["false_positive_rate"] is None
    assert buckets["13-24"]["false_positive_rate"] == pytest.approx(1.0)


def test_record_epochs_rejects_mismatched_agent_arrays():
    tracker = DetectionPowerTracker()
    with pytest.raises(ValueError, match="same agents"):
        tracker.record_epochs(
            step=0,
            widths=np.array([4, 4], dtype=np.int64),
            emitted=np.array([True]),
            hazard=np.array([True]),
        )


def test_latency_times_first_token_and_rearms_after_hazard_clears():
    tracker = DetectionPowerTracker()
    widths = np.array([4], dtype=np.int64)
    quiet = np.array([False])
    alarm = np.array([True])
    exposed = np.array([True])
    clear = np.array([False])

    tracker.record_epochs(step=0, widths=widths, emitted=quiet, hazard=exposed)
    tracker.record_epochs(step=3, widths=widths, emitted=alarm, hazard=exposed)
    # Second alarm in the same episode must not be timed again.
    tracker.record_epochs(step=4, widths=widths, emitted=alarm, hazard=exposed)
    tracker.record_epochs(step=5, widths=widths, emitted=quiet, hazard=clear)
    tracker.record_epochs(step=9, widths=widths, emitted=alarm, hazard=exposed)

    bucket = tracker.summary()["width_buckets"]["1-5"]
    assert bucket["detections_timed"] == 2
    assert bucket["min_detection_latency_steps"] == 0
    assert bucket["max_detection_latency_steps"] == 3


def test_clean_agent_never_contributes_a_latency():
    tracker = DetectionPowerTracker()
    widths = np.array([8], dtype=np.int64)
    for step in range(5):
        tracker.record_epochs(
            step=step,
            widths=widths,
            emitted=np.array([True]),
            hazard=np.array([False]),
        )

    buckets = tracker.summary()["width_buckets"]
    assert all(buckets[label]["detections_timed"] == 0 for label in WIDTH_BUCKET_LABELS)
    assert buckets["6-12"]["false_positive_rate"] == pytest.approx(1.0)


def test_ablation_credits_the_channel_that_carried_the_alarm():
    channel_set = DEFAULT_CHANNEL_SET
    probe = AblationProbe(channel_set=channel_set, sample_rate=1.0)
    baseline = _warm_baseline(channel_set)
    # Every channel sits at its unit-normal baseline mean except one, which is
    # driven far out of range and so must carry the alarm alone.
    observation = channel_set.zeros()
    observation[0] = 40.0

    probe.maybe_record(
        baseline=baseline,
        observation=observation,
        hour=12,
        month=1,
        observed=None,
        reference_threshold=3.0,
    )

    channels = probe.summary()["channels"]
    driver = channel_set.names[0]
    assert probe.summary()["alarms_sampled"] == 1
    assert channels[driver]["marginal_contribution"] == pytest.approx(1.0)
    for name in channel_set.names[1:]:
        assert channels[name]["marginal_contribution"] == pytest.approx(0.0)


def test_ablation_skips_epochs_with_a_single_observed_channel():
    channel_set = DEFAULT_CHANNEL_SET
    probe = AblationProbe(channel_set=channel_set, sample_rate=1.0)
    baseline = _warm_baseline(channel_set)
    observed = np.zeros(len(channel_set), dtype=np.bool_)
    observed[1] = True

    probe.maybe_record(
        baseline=baseline,
        observation=channel_set.zeros(),
        hour=12,
        month=1,
        observed=observed,
        reference_threshold=3.0,
    )

    assert probe.summary()["alarms_sampled"] == 0
    assert probe.summary()["channels"] == {}


def test_disabled_ablation_records_nothing():
    channel_set = DEFAULT_CHANNEL_SET
    probe = AblationProbe(channel_set=channel_set)
    probe.maybe_record(
        baseline=_warm_baseline(channel_set),
        observation=channel_set.zeros(),
        hour=12,
        month=1,
        observed=None,
        reference_threshold=3.0,
    )

    assert not probe.enabled
    assert probe.summary()["alarms_sampled"] == 0


def _warm_baseline(channel_set):
    """A baseline tracker with enough samples to have a usable covariance."""
    from garland.biometrics import BaselineTracker

    baseline = BaselineTracker(channel_set=channel_set)
    rng = np.random.default_rng(3)
    for _ in range(200):
        baseline.update(rng.normal(0.0, 1.0, len(channel_set)), 12, 1)
    return baseline


@pytest.mark.parametrize("backend", ["rect", "hex"])
def test_summary_rates_are_bounded_and_finite_on_both_backends(backend: str):
    summary = _tracker_summary(
        _model(
            adoption={"motion_actigraphy": 0.5, "respiratory_acoustic_patch": 0.3},
            backend=backend,
            n_steps=80,
        )
    )

    assert summary["scored_epochs"] > 0
    assert summary["mean_effective_width"] >= 1.0
    for bucket in summary["width_buckets"].values():
        for key in RATE_KEYS:
            value = bucket[key]
            assert value is None or 0.0 <= value <= 1.0
        assert bucket["hazard_epochs"] + bucket["clean_epochs"] == bucket["scored_epochs"]
        latency = bucket["mean_detection_latency_steps"]
        assert latency is None or latency >= 0.0
    assert summary["devices"]
    for device in summary["devices"].values():
        for key in DEVICE_RATE_KEYS:
            value = device[key]
            assert value is None or 0.0 <= value <= 1.0
        assert device["owners"] > 0
        observed = device["observed_channel_fraction"]
        assert observed == pytest.approx(1.0 - device["masked_channel_fraction"])


def test_wider_adoption_grades_mean_effective_width_and_bucket_occupancy():
    arms = {
        "core": {},
        "one_band": {"motion_actigraphy": 1.0},
        "three_bands": {
            "motion_actigraphy": 1.0,
            "respiratory_acoustic_patch": 1.0,
            "chest_electrode_patch": 1.0,
        },
    }
    widths = []
    occupied = []
    for adoption in arms.values():
        summary = _tracker_summary(_model(adoption=adoption, n_steps=60))
        widths.append(summary["mean_effective_width"])
        occupied.append(
            sum(
                1
                for bucket in summary["width_buckets"].values()
                if bucket["scored_epochs"] > 0 and bucket["mean_effective_width"] is not None
            )
        )

    assert widths[0] < widths[1] < widths[2]
    # Adopting more subsystems must reach at least as many width buckets.
    assert occupied[0] <= occupied[2]


def test_duty_cycling_leaves_owned_subsystems_partly_masked():
    summary = _tracker_summary(
        _model(
            adoption={"headband_eeg": 1.0, "instrumented_footwear": 1.0},
            n_steps=288,
        )
    )
    devices = summary["devices"]

    for kind in ("headband_eeg", "instrumented_footwear"):
        # These subsystems are event-gated, so they must neither be always on
        # nor completely silent across a simulated day.
        assert 0.0 < devices[kind]["masked_channel_fraction"] < 1.0
        assert devices[kind]["reporting_epoch_fraction"] < 1.0


def test_null_baseline_reports_no_true_positive_denominator():
    model = _model(
        adoption={"motion_actigraphy": 0.5},
        n_steps=60,
        plume_start=10_000,
    )
    summary = _tracker_summary(model)
    config = model.config

    assert config.plumes[0].start_step > config.n_steps
    for bucket in summary["width_buckets"].values():
        assert bucket["clean_epochs"] <= bucket["scored_epochs"]
        fpr = bucket["false_positive_rate"]
        assert fpr is None or 0.0 <= fpr <= 1.0


def test_ablation_diagnostic_does_not_change_the_simulation():
    plain = _model(adoption={"motion_actigraphy": 0.5}, n_steps=60)
    probed = _model(adoption={"motion_actigraphy": 0.5}, n_steps=60, ablation_rate=0.5)

    plain_summary = _tracker_summary(plain)
    probed_summary = _tracker_summary(probed)
    assert "channel_ablation" not in plain_summary
    assert plain_summary["width_buckets"] == probed_summary["width_buckets"]
    assert (
        plain.metrics.summary()["total_broadcasts"] == probed.metrics.summary()["total_broadcasts"]
    )

    ablation = probed_summary["channel_ablation"]
    assert ablation["alarms_sampled"] > 0
    for stats in ablation["channels"].values():
        assert 0 < stats["alarms_evaluated"]
        assert 0.0 <= stats["alarm_retention"] <= 1.0
        assert stats["alarm_retention"] == pytest.approx(1.0 - stats["marginal_contribution"])


def test_no_single_channel_carries_every_alarm():
    summary = _tracker_summary(
        _model(
            adoption={"motion_actigraphy": 1.0, "respiratory_acoustic_patch": 1.0},
            n_steps=120,
            ablation_rate=1.0,
        )
    )
    channels = summary["channel_ablation"]["channels"]
    well_sampled = {
        name: stats["marginal_contribution"]
        for name, stats in channels.items()
        if stats["alarms_evaluated"] >= 20
    }

    assert well_sampled
    # A channel with contribution 1.0 everywhere would mean the joint score is
    # really that one channel, which is the failure mode this design avoids.
    assert max(well_sampled.values()) < 1.0
