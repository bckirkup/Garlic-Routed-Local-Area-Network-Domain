"""Regression coverage for the wrist SpO2, skin-temperature, and EDA channels."""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import CORE_VITALS, EDA_SCL, ChannelSet
from garland.demographics import AGE_BANDS, DemographicsConfig
from garland.devices import (
    WRIST_EDA_MODULE,
    WRIST_PPG,
    DeviceFleet,
    DeviceFleetConfig,
    build_channel_set,
)
from garland.modality_signatures import (
    heat_strain_axes,
    infection_axes,
    irritant_axes,
    modality_delta,
)


def test_wrist_channels_are_fleet_only_not_core_vitals() -> None:
    channel_set = build_channel_set((WRIST_PPG, WRIST_EDA_MODULE))

    assert channel_set.names[: len(CORE_VITALS)] == CORE_VITALS.names
    assert set(("spo2_pct", "wrist_skin_temperature", "eda_scl_microsiemens")) <= set(
        channel_set.names
    )
    assert not CORE_VITALS.has("spo2_pct")
    assert not CORE_VITALS.has("wrist_skin_temperature")
    assert not CORE_VITALS.has("eda_scl_microsiemens")


def test_new_modalities_follow_distinct_illness_axes() -> None:
    channel_set = build_channel_set((WRIST_PPG, WRIST_EDA_MODULE))
    infection = modality_delta(infection_axes(1.0), channel_set)
    irritant = modality_delta(irritant_axes(1.0), channel_set)
    heat = modality_delta(heat_strain_axes(1.0), channel_set)

    spo2 = channel_set.index("spo2_pct")
    skin = channel_set.index("wrist_skin_temperature")
    eda = channel_set.index("eda_scl_microsiemens")
    assert infection[spo2] < 0.0
    assert infection[skin] > 0.0
    assert infection[eda] > 0.0
    assert irritant[spo2] < 0.0
    assert irritant[skin] == pytest.approx(-0.3)
    assert heat[spo2] == pytest.approx(0.0)
    assert heat[skin] > 0.0


def test_eda_ownership_draw_respects_age_affinity() -> None:
    n_wearable = 10_000
    age_bands = np.repeat(np.arange(len(AGE_BANDS), dtype=np.int8), n_wearable // len(AGE_BANDS))
    fleet = DeviceFleet(
        n_wearable=len(age_bands),
        config=DeviceFleetConfig(
            enabled=True,
            adoption={"wrist_eda_module": 0.5},
        ),
        rng=np.random.default_rng(42),
        age_bands=age_bands,
        demographics=DemographicsConfig(enabled=True, enthusiasm_sigma=0.0),
    )
    eda_position = next(
        position for position, kind in enumerate(fleet.kinds) if kind.name == "wrist_eda_module"
    )
    owned = fleet.ownership[:, eda_position]
    rates = [float(np.mean(owned[age_bands == index])) for index in range(len(AGE_BANDS))]

    assert int(np.count_nonzero(owned)) == int(np.floor(0.5 * n_wearable))
    assert rates[2] > rates[0]
    assert rates[2] > rates[1]


def test_eda_hard_floor_survives_negative_delta() -> None:
    channel_set = ChannelSet((EDA_SCL,))
    clamped = channel_set.clamp(np.array([-10.0]))

    assert clamped[0] == pytest.approx(EDA_SCL.floor)
