"""Behavioral tests for host-stratified detection-power metrics."""

from __future__ import annotations

import numpy as np
import pytest

from garland.detection_power import DetectionPowerTracker
from garland.host_phenotypes import HostPhenotypeConfig
from garland.simulation import GarlandModel, SimulationConfig


def _summary(
    *,
    groups: dict[str, np.ndarray],
    widths: np.ndarray,
    emitted: np.ndarray,
    infected: np.ndarray,
    exposed: np.ndarray,
    step: int = 0,
    tracker: DetectionPowerTracker | None = None,
) -> dict:
    tracker = tracker or DetectionPowerTracker()
    tracker.record_host_group_epochs(
        step=step,
        groups=groups,
        widths=widths,
        emitted=emitted,
        infected=infected,
        exposed=exposed,
    )
    return tracker.summary()["host_groups"]


def test_group_true_positive_rate_responds_to_group_emissions() -> None:
    groups = {
        "a": np.array([True, False]),
        "b": np.array([False, True]),
    }
    summary = _summary(
        groups=groups,
        widths=np.array([4, 4]),
        emitted=np.array([True, False]),
        infected=np.array([True, True]),
        exposed=np.zeros(2, dtype=bool),
    )

    assert summary["a"]["disease"]["true_positive_rate"] > 0.0
    assert summary["b"]["disease"]["true_positive_rate"] == pytest.approx(0.0)


def test_overlapping_groups_each_receive_hazard_and_latency() -> None:
    tracker = DetectionPowerTracker()
    groups = {
        "a": np.array([True]),
        "b": np.array([True]),
    }
    widths = np.array([4])
    tracker.record_host_group_epochs(
        step=2,
        groups=groups,
        widths=widths,
        emitted=np.array([False]),
        infected=np.array([True]),
        exposed=np.array([False]),
    )
    tracker.record_host_group_epochs(
        step=5,
        groups=groups,
        widths=widths,
        emitted=np.array([True]),
        infected=np.array([True]),
        exposed=np.array([False]),
    )

    summary = tracker.summary()["host_groups"]
    for group in groups:
        assert summary[group]["disease"]["detections_timed"] == 1
        assert summary[group]["disease"]["mean_detection_latency_steps"] == pytest.approx(3.0)


def test_general_group_excludes_flagged_agents() -> None:
    groups = {
        "diabetic": np.array([True, False, False]),
        "general": np.array([False, True, False]),
    }
    summary = _summary(
        groups=groups,
        widths=np.array([4, 4, 4]),
        emitted=np.zeros(3, dtype=bool),
        infected=np.zeros(3, dtype=bool),
        exposed=np.zeros(3, dtype=bool),
    )

    assert summary["diabetic"]["owners"] == 1
    assert summary["general"]["owners"] == 1
    assert summary["general"]["scored_epochs"] == 1


def test_latency_rearms_after_hazard_clears() -> None:
    tracker = DetectionPowerTracker()
    groups = {"general": np.array([True])}
    widths = np.array([4])
    tracker.record_host_group_epochs(
        step=1,
        groups=groups,
        widths=widths,
        emitted=np.array([False]),
        infected=np.array([True]),
        exposed=np.array([False]),
    )
    tracker.record_host_group_epochs(
        step=4,
        groups=groups,
        widths=widths,
        emitted=np.array([True]),
        infected=np.array([True]),
        exposed=np.array([False]),
    )
    tracker.record_host_group_epochs(
        step=5,
        groups=groups,
        widths=widths,
        emitted=np.array([False]),
        infected=np.array([False]),
        exposed=np.array([False]),
    )
    tracker.record_host_group_epochs(
        step=9,
        groups=groups,
        widths=widths,
        emitted=np.array([True]),
        infected=np.array([True]),
        exposed=np.array([False]),
    )

    disease = tracker.summary()["host_groups"]["general"]["disease"]
    assert disease["detections_timed"] == 2
    assert disease["min_detection_latency_steps"] == 0
    assert disease["max_detection_latency_steps"] == 3


def test_infected_and_exposed_epoch_counts_in_both_hazard_cells() -> None:
    summary = _summary(
        groups={"general": np.array([True])},
        widths=np.array([4]),
        emitted=np.array([True]),
        infected=np.array([True]),
        exposed=np.array([True]),
    )

    group = summary["general"]
    assert group["disease"]["hazard_epochs"] == 1
    assert group["toxin"]["hazard_epochs"] == 1
    assert group["clean_epochs"] == 0


def test_non_reporting_epochs_are_excluded_from_group_denominators() -> None:
    summary = _summary(
        groups={"general": np.array([True, True])},
        widths=np.array([0, 4]),
        emitted=np.array([True, False]),
        infected=np.array([True, False]),
        exposed=np.zeros(2, dtype=bool),
    )

    group = summary["general"]
    assert group["owners"] == 2
    assert group["scored_epochs"] == 1
    assert group["clean_epochs"] == 1
    assert group["disease"]["hazard_epochs"] == 0


def test_tracker_has_no_host_groups_until_a_group_is_registered() -> None:
    assert "host_groups" not in DetectionPowerTracker().summary()


def test_hosts_disabled_model_keeps_detection_power_unstratified() -> None:
    config = SimulationConfig(
        n_agents=80,
        wearable_fraction=0.25,
        n_steps=2,
        seed=42,
        spatial_backend="rect",
        mobility_model="static",
        world_settling_steps=0,
    )
    summary = GarlandModel(config).run().summary()["detection_power"]

    assert config.hosts.enabled is False
    assert "host_groups" not in summary


def test_hosts_enabled_model_registers_all_host_groups() -> None:
    config = SimulationConfig(
        n_agents=120,
        wearable_fraction=0.25,
        n_steps=2,
        seed=42,
        spatial_backend="rect",
        mobility_model="static",
        world_settling_steps=0,
        hosts=HostPhenotypeConfig(enabled=True),
    )
    summary = GarlandModel(config).run().summary()["detection_power"]

    assert set(summary["host_groups"]) == {
        "diabetic",
        "frail_elderly",
        "law_enforcement",
        "assistive_need",
        "general",
    }
