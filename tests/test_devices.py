"""Tests for per-modality device bundles, adoption, and duty-cycle masks."""

from __future__ import annotations

import numpy as np
import pytest

from garland.app import build_config_from_args, parse_run_args
from garland.channels import CORE_VITALS
from garland.config import config_from_dict, config_to_dict
from garland.devices import (
    ABDOMINAL_ACOUSTIC_BAND,
    BASE_DEVICE_KIND,
    DEVICE_CATALOGUE,
    THORACIC_EIT_ACOUSTIC_BAND,
    DeviceChannel,
    DeviceFleet,
    DeviceFleetConfig,
    build_channel_set,
)
from garland.simulation import GarlandModel, SimulationConfig

ALL_MODALITIES = {
    THORACIC_EIT_ACOUSTIC_BAND.name: 0.5,
    ABDOMINAL_ACOUSTIC_BAND.name: 0.25,
}

BAND_KINDS = (BASE_DEVICE_KIND, THORACIC_EIT_ACOUSTIC_BAND, ABDOMINAL_ACOUSTIC_BAND)


def make_fleet(
    n_wearable: int = 400,
    adoption: dict[str, float] | None = None,
    seed: int = 3,
) -> DeviceFleet:
    config = DeviceFleetConfig(
        enabled=True, adoption=dict(ALL_MODALITIES if adoption is None else adoption)
    )
    return DeviceFleet(n_wearable, config, np.random.default_rng(seed))


def observed_fraction(
    fleet: DeviceFleet,
    channel_name: str,
    *,
    hour_of_day: float,
    activity_level: float,
    n_epochs: int = 60,
    seed: int = 11,
) -> float:
    """Fraction of owner-epochs in which ``channel_name`` was reported."""
    rng = np.random.default_rng(seed)
    column = fleet.channel_set.index(channel_name)
    owner_column = next(
        position
        for position, kind in enumerate(fleet.kinds)
        if channel_name in {channel.name for channel in kind.channels}
    )
    owners = fleet.ownership[:, owner_column]
    reported = 0
    for _ in range(n_epochs):
        matrix = fleet.observed_matrix(hour_of_day, activity_level, rng)
        reported += int(np.count_nonzero(matrix[owners, column]))
    return reported / (n_epochs * int(np.count_nonzero(owners)))


def test_channel_set_starts_with_core_vitals_and_is_stable() -> None:
    kinds = (BASE_DEVICE_KIND, THORACIC_EIT_ACOUSTIC_BAND, ABDOMINAL_ACOUSTIC_BAND)
    channel_set = build_channel_set(kinds)
    assert channel_set.names[: len(CORE_VITALS)] == CORE_VITALS.names
    assert build_channel_set(kinds).names == channel_set.names
    assert len(set(channel_set.names)) == len(channel_set)


def test_every_catalogued_channel_appears_exactly_once() -> None:
    channel_set = build_channel_set(tuple(DEVICE_CATALOGUE.values()))
    for kind in DEVICE_CATALOGUE.values():
        for channel in kind.channels:
            assert channel_set.has(channel.name)
    assert len(channel_set) == len(set(channel_set.names))


def test_disabled_fleet_keeps_core_vitals_only() -> None:
    fleet = DeviceFleet(50, DeviceFleetConfig(), np.random.default_rng(0))
    assert fleet.channel_set.names == CORE_VITALS.names
    assert fleet.owner_counts() == {BASE_DEVICE_KIND.name: 50}


def test_unstructured_ownership_keeps_numpy_choice_sampling() -> None:
    n_wearable = 100
    n_owners = 30
    seed = 17
    fleet = DeviceFleet(
        n_wearable,
        DeviceFleetConfig(
            enabled=True,
            adoption={THORACIC_EIT_ACOUSTIC_BAND.name: n_owners / n_wearable},
        ),
        np.random.default_rng(seed),
    )
    position = fleet.kinds.index(THORACIC_EIT_ACOUSTIC_BAND)
    owners = np.flatnonzero(fleet.ownership[:, position])
    expected = np.sort(np.random.default_rng(seed).choice(n_wearable, size=n_owners, replace=False))
    assert np.array_equal(owners, expected)


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_adoption_count_tracks_configured_fraction(fraction: float) -> None:
    n_wearable = 500
    fleet = make_fleet(n_wearable, {THORACIC_EIT_ACOUSTIC_BAND.name: fraction})
    owners = fleet.owner_counts()[THORACIC_EIT_ACOUSTIC_BAND.name]
    assert owners == pytest.approx(fraction * n_wearable, abs=1)
    assert fleet.owner_counts()[BASE_DEVICE_KIND.name] == n_wearable


def test_modality_ownership_is_independent_across_kinds() -> None:
    fleet = make_fleet(2000)
    thoracic = fleet.ownership[:, fleet.kinds.index(THORACIC_EIT_ACOUSTIC_BAND)]
    abdominal = fleet.ownership[:, fleet.kinds.index(ABDOMINAL_ACOUSTIC_BAND)]
    both = float(np.count_nonzero(thoracic & abdominal)) / fleet.n_wearable
    # Independent sampling puts the overlap near the product of the fractions;
    # perfect nesting (0.25) or disjointness (0.0) would both fail this.
    assert both == pytest.approx(0.5 * 0.25, abs=0.05)


def test_unknown_device_kind_is_rejected() -> None:
    fleet = DeviceFleetConfig(enabled=True, adoption={"ankle_barometer": 0.5})
    with pytest.raises(ValueError, match="unknown device kind"):
        fleet.resolved_adoption()


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_out_of_range_adoption_is_rejected(fraction: float) -> None:
    config = DeviceFleetConfig(enabled=True, adoption={ABDOMINAL_ACOUSTIC_BAND.name: fraction})
    with pytest.raises(ValueError, match="must be in"):
        config.resolved_adoption()


def test_out_of_range_duty_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="duty_cycle"):
        DeviceChannel(channel=CORE_VITALS.channels[0], duty_cycle=1.4)


def test_non_owners_never_report_modality_channels() -> None:
    """Negative control: a channel is not reported without the device for it."""
    fleet = make_fleet(300)
    rng = np.random.default_rng(5)
    for kind in (THORACIC_EIT_ACOUSTIC_BAND, ABDOMINAL_ACOUSTIC_BAND):
        non_owners = ~fleet.ownership[:, fleet.kinds.index(kind)]
        columns = [fleet.channel_set.index(channel.name) for channel in kind.channels]
        for _ in range(20):
            matrix = fleet.observed_matrix(14.0, 0.2, rng)
            assert not matrix[np.ix_(non_owners, columns)].any()
    # And core vitals are unaffected by modality ownership.
    matrix = fleet.observed_matrix(14.0, 0.2, rng)
    core_columns = [fleet.channel_set.index(name) for name in CORE_VITALS.names]
    assert matrix[:, core_columns].all()


def test_mask_shape_and_dtype_are_invariant() -> None:
    fleet = make_fleet(120)
    rng = np.random.default_rng(9)
    for hour in (0.0, 5.5, 11.0, 16.0, 22.0, 23.9):
        matrix = fleet.observed_matrix(hour, 0.5, rng)
        assert matrix.shape == (fleet.n_wearable, len(fleet.channel_set))
        assert matrix.dtype == np.bool_
        # Core vitals are always reported, so no owner is ever fully missing.
        assert matrix.any(axis=1).all()


@pytest.mark.parametrize(
    ("kind", "channel_name"),
    [
        (THORACIC_EIT_ACOUSTIC_BAND, "regional_ventilation_heterogeneity"),
        (THORACIC_EIT_ACOUSTIC_BAND, "pep_ms"),
        (THORACIC_EIT_ACOUSTIC_BAND, "pwv_m_s"),
        (THORACIC_EIT_ACOUSTIC_BAND, "eit_perfusion_pulsatility_ratio"),
        (ABDOMINAL_ACOUSTIC_BAND, "bowel_sound_burst_rate"),
        (ABDOMINAL_ACOUSTIC_BAND, "acoustic_motility_index"),
        (ABDOMINAL_ACOUSTIC_BAND, "bladder_filling_impedance_shift"),
    ],
)
def test_resting_yield_is_within_calibrated_band(kind, channel_name: str) -> None:
    """Sedentary daytime yield sits at the device's configured duty cycle."""
    fleet = make_fleet(300, {kind.name: 1.0})
    binding = next(b for b in kind.device_channels if b.channel.name == channel_name)
    fraction = observed_fraction(fleet, channel_name, hour_of_day=14.0, activity_level=0.0)
    assert fraction == pytest.approx(binding.duty_cycle, abs=0.03)
    assert 0.0 < fraction < 1.0


def test_yield_degrades_monotonically_with_activity() -> None:
    """Graded sensitivity: more motion, strictly less usable acoustic data."""
    fleet = make_fleet(300, {ABDOMINAL_ACOUSTIC_BAND.name: 1.0})
    fractions = [
        observed_fraction(
            fleet, "bowel_sound_burst_rate", hour_of_day=14.0, activity_level=activity
        )
        for activity in (0.0, 0.3, 0.6, 1.0)
    ]
    assert fractions == sorted(fractions, reverse=True)
    assert fractions[0] - fractions[-1] > 0.2
    assert all(0.0 <= fraction <= 1.0 for fraction in fractions)


@pytest.mark.parametrize("channel_name", ["bowel_sound_burst_rate", "acoustic_motility_index"])
def test_abdominal_acoustics_yield_more_during_sleep(channel_name: str) -> None:
    fleet = make_fleet(300, {ABDOMINAL_ACOUSTIC_BAND.name: 1.0})
    awake = observed_fraction(fleet, channel_name, hour_of_day=14.0, activity_level=0.1)
    asleep = observed_fraction(fleet, channel_name, hour_of_day=2.0, activity_level=0.1)
    assert asleep > awake + 0.05


def test_pelvic_impedance_outlasts_abdominal_acoustics_while_moving() -> None:
    """Impedance survives motion that buries a contact microphone."""
    fleet = make_fleet(300, {ABDOMINAL_ACOUSTIC_BAND.name: 1.0})
    bladder = observed_fraction(
        fleet, "bladder_filling_impedance_shift", hour_of_day=14.0, activity_level=0.8
    )
    bowel = observed_fraction(fleet, "bowel_sound_burst_rate", hour_of_day=14.0, activity_level=0.8)
    assert bladder > bowel


def test_gastric_channel_reports_only_at_event_completion() -> None:
    fleet = make_fleet(200, {ABDOMINAL_ACOUSTIC_BAND.name: 1.0})
    column = fleet.channel_set.index("gastric_emptying_index")
    rng = np.random.default_rng(4)
    completions = next(
        binding.event_completion_hours
        for binding in ABDOMINAL_ACOUSTIC_BAND.device_channels
        if binding.channel.name == "gastric_emptying_index"
    )
    for hour in (0.0, 6.0, 9.0, 13.0, 18.5):
        assert hour not in completions
        assert not fleet.observed_matrix(hour, 0.1, rng)[:, column].any()
    for hour in completions:
        reported = fleet.observed_matrix(hour, 0.1, rng)[:, column]
        assert 0 < int(np.count_nonzero(reported)) < fleet.n_wearable


def test_gastric_reports_a_few_times_per_day() -> None:
    """Over a simulated day the event-gated channel reports only a few epochs."""
    fleet = make_fleet(50, {ABDOMINAL_ACOUSTIC_BAND.name: 1.0})
    column = fleet.channel_set.index("gastric_emptying_index")
    rng = np.random.default_rng(6)
    reporting_epochs = 0
    for step in range(288):
        hour = step * 24.0 / 288.0
        if fleet.observed_matrix(hour, 0.1, rng)[:, column].any():
            reporting_epochs += 1
    assert reporting_epochs == 3


def test_simulation_widens_layout_and_masks_unowned_channels() -> None:
    config = SimulationConfig(
        n_agents=400,
        n_steps=30,
        wearable_fraction=0.3,
        seed=13,
        devices=DeviceFleetConfig(enabled=True, adoption=dict(ALL_MODALITIES)),
    )
    model = GarlandModel(config)
    assert model.device_fleet is not None
    assert len(model.channel_set) == len(build_channel_set(BAND_KINDS))
    assert model.channel_set.names[: len(CORE_VITALS)] == CORE_VITALS.names
    for agent in model.citizen_agents:
        assert len(agent.baseline.ema) == len(model.channel_set)
    for _ in range(30):
        model.step()
    for agent in model.citizen_agents:
        assert np.all(np.isfinite(agent.baseline.ema))
        assert np.all(np.isfinite(agent.baseline.covariance_matrix()))


@pytest.mark.parametrize("backend", ["hex", "rect"])
def test_device_fleet_runs_on_both_spatial_backends(backend: str) -> None:
    config = SimulationConfig(
        n_agents=300,
        n_steps=20,
        wearable_fraction=0.3,
        seed=21,
        spatial_backend=backend,
        devices=DeviceFleetConfig(enabled=True, adoption=dict(ALL_MODALITIES)),
    )
    model = GarlandModel(config)
    for _ in range(20):
        model.step()
    assert model.device_fleet is not None
    assert model.current_step == 20
    for agent in model.citizen_agents:
        assert np.all(np.isfinite(agent.baseline.ema))


def test_default_config_leaves_simulation_unwidened() -> None:
    model = GarlandModel(SimulationConfig(n_agents=200, n_steps=5, seed=2))
    assert model.device_fleet is None
    assert model.channel_set.names == CORE_VITALS.names


def test_config_round_trip_preserves_device_adoption() -> None:
    config = config_from_dict(
        {
            "n_agents": 100,
            "devices": {"enabled": True, "adoption": {THORACIC_EIT_ACOUSTIC_BAND.name: 0.4}},
        }
    )
    assert config.devices.enabled
    assert config.devices.adoption == {THORACIC_EIT_ACOUSTIC_BAND.name: 0.4}
    restored = config_from_dict(config_to_dict(config))
    assert restored.devices == config.devices


def test_config_rejects_unknown_device_kind_at_load() -> None:
    with pytest.raises(ValueError, match="unknown device kind"):
        config_from_dict({"devices": {"enabled": True, "adoption": {"nose_ring": 0.1}}})


def test_config_without_devices_section_stays_disabled() -> None:
    config = config_from_dict({"n_agents": 100})
    assert not config.devices.enabled
    assert config.devices.adoption == {}


def test_cli_device_adoption_flag_enables_the_fleet() -> None:
    config = build_config_from_args(
        parse_run_args(
            ["--device-adoption", f"{ABDOMINAL_ACOUSTIC_BAND.name}=0.2", "--n-agents", "100"]
        )
    )
    assert config.devices.enabled
    assert config.devices.adoption == {ABDOMINAL_ACOUSTIC_BAND.name: 0.2}


def test_cli_without_device_flag_leaves_fleet_disabled() -> None:
    config = build_config_from_args(parse_run_args(["--n-agents", "100"]))
    assert not config.devices.enabled
