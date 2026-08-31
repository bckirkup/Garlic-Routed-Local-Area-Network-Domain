"""Behavioral coverage for CGM sustained-glucose detection."""

from __future__ import annotations

import numpy as np
import pytest

from garland.agents import CitizenAgent
from garland.biometric_profiles import build_profile
from garland.channels import INTERSTITIAL_GLUCOSE, ChannelSet
from garland.config import config_from_dict, config_to_dict
from garland.detection import GlucoseCusum
from garland.devices import DeviceFleetConfig
from garland.host_phenotypes import HostPhenotypeConfig
from garland.perturbations import PerturbationCause, PerturbationContribution
from garland.privacy import AnomalyType
from garland.simulation import GarlandModel, SimulationConfig


def test_sustained_residual_latches_once() -> None:
    detector = GlucoseCusum()
    alarms = [detector.update(2.0) for _ in range(40)]

    assert sum(alarms) == 1
    assert detector.alarm_active
    assert detector.statistic > detector.threshold


def test_meal_pulses_do_not_latch() -> None:
    detector = GlucoseCusum()
    rise = np.linspace(0.0, 80.0 / 25.0, 6)
    decay = np.linspace(80.0 / 25.0, 0.0, 24)[1:]
    pulse = np.concatenate((rise, decay, np.zeros(60)))

    alarms = [detector.update(float(value)) for _ in range(3) for value in pulse]

    assert not any(alarms)
    assert not detector.alarm_active
    assert detector.statistic == pytest.approx(0.0)


def test_negative_residuals_cannot_accumulate() -> None:
    detector = GlucoseCusum()

    for _ in range(100):
        detector.update(-3.0)

    assert detector.statistic == pytest.approx(0.0)
    assert not detector.alarm_active


def test_alarm_clears_after_low_statistic_epochs() -> None:
    detector = GlucoseCusum(threshold=4.0, clear_steps=3)
    assert any(detector.update(2.0) for _ in range(100))
    assert detector.alarm_active

    detector.update(0.0)
    assert detector.alarm_active
    detector.update(0.0)
    assert detector.statistic <= detector.threshold * detector.clear_fraction
    detector.update(0.0)
    assert detector.alarm_active
    detector.update(0.0)
    assert not detector.alarm_active

    assert detector.statistic == pytest.approx(0.0)
    assert detector.zero_steps == 0
    assert not detector.alarm_active


def _glucose_agent(detector: GlucoseCusum | None) -> tuple[CitizenAgent, int]:
    channel_set = ChannelSet((INTERSTITIAL_GLUCOSE,))
    profile = build_profile(channel_set=channel_set)
    agent = CitizenAgent(
        idx=0,
        has_wearable=True,
        profile=profile,
        anomaly_threshold=10.0,
        glucose_detector=detector,
    )
    return agent, channel_set.index(INTERSTITIAL_GLUCOSE.name)


def _glucose_perturbation(channel_set: ChannelSet, amount: float = 50.0):
    return (
        PerturbationContribution(
            PerturbationCause.DISEASE,
            channel_set.delta({INTERSTITIAL_GLUCOSE.name: amount}),
        ),
    )


def test_agent_emits_multisystem_for_sustained_glucose() -> None:
    detector = GlucoseCusum(slack_sd=0.0, threshold=1.5)
    agent, glucose_index = _glucose_agent(detector)
    rng = np.random.default_rng(8)
    instant_token = agent.observe_and_detect(
        hour=12,
        month=1,
        day_of_year=15,
        hour_of_day=12.0,
        rng=rng,
        cell_id=3,
        perturbations=_glucose_perturbation(agent.channel_set),
    )
    assert instant_token is None
    assert agent.last_observation[glucose_index] > 135.0

    token = None
    for _ in range(20):
        token = agent.observe_and_detect(
            hour=12,
            month=1,
            day_of_year=15,
            hour_of_day=12.0,
            rng=rng,
            cell_id=3,
            perturbations=_glucose_perturbation(agent.channel_set),
        )
        if token is not None:
            break

    assert token is not None
    assert token.anomaly_type == AnomalyType.MULTI_SYSTEM
    assert agent.anomaly_active


def test_masked_glucose_freezes_cusum() -> None:
    detector = GlucoseCusum()
    agent, glucose_index = _glucose_agent(detector)
    rng = np.random.default_rng(9)
    observed = np.ones(len(agent.channel_set), dtype=np.bool_)
    observed[glucose_index] = False

    for _ in range(10):
        assert (
            agent.observe_and_detect(
                hour=12,
                month=1,
                day_of_year=15,
                hour_of_day=12.0,
                rng=rng,
                cell_id=3,
                perturbations=_glucose_perturbation(agent.channel_set),
                observed_channels=observed,
            )
            is None
        )

    assert detector.statistic == pytest.approx(0.0)
    assert not detector.alarm_active


def test_suppression_resets_glucose_cusum() -> None:
    detector = GlucoseCusum()
    detector.statistic = 10.0
    detector.alarm_active = True
    agent, _ = _glucose_agent(detector)

    agent.observe_and_detect(
        hour=12,
        month=1,
        day_of_year=15,
        hour_of_day=12.0,
        rng=np.random.default_rng(10),
        cell_id=3,
        suppress_token_emission=True,
    )

    assert detector.statistic == pytest.approx(0.0)
    assert not detector.alarm_active


def test_agent_without_glucose_detector_preserves_no_token_path() -> None:
    agent, _ = _glucose_agent(None)

    token = agent.observe_and_detect(
        hour=12,
        month=1,
        day_of_year=15,
        hour_of_day=12.0,
        rng=np.random.default_rng(11),
        cell_id=3,
        perturbations=_glucose_perturbation(agent.channel_set),
    )

    assert token is None
    assert not agent.anomaly_active


def test_config_round_trip_and_model_wiring() -> None:
    config = SimulationConfig(
        n_agents=100,
        n_steps=1,
        wearable_fraction=0.5,
        hosts=HostPhenotypeConfig(enabled=True),
        devices=DeviceFleetConfig(enabled=True, adoption={"cgm_patch": 0.5}),
        glucose_cusum_slack_sd=1.1,
        glucose_cusum_threshold=9.0,
        glucose_cusum_clear_steps=4,
    )
    restored = config_from_dict(config_to_dict(config))
    assert restored.glucose_cusum_enabled is True
    assert restored.glucose_cusum_slack_sd == pytest.approx(1.1)
    assert restored.glucose_cusum_threshold == pytest.approx(9.0)
    assert restored.glucose_cusum_clear_steps == 4

    model = GarlandModel(config)
    assert any(agent.glucose_detector is not None for agent in model.citizen_agents)

    no_cgm = GarlandModel(
        SimulationConfig(
            n_agents=40,
            n_steps=1,
            wearable_fraction=0.5,
            hosts=HostPhenotypeConfig(enabled=True),
        )
    )
    assert all(agent.glucose_detector is None for agent in no_cgm.citizen_agents)

    disabled = GarlandModel(
        SimulationConfig(
            n_agents=40,
            n_steps=1,
            wearable_fraction=0.5,
            glucose_cusum_enabled=False,
        )
    )
    assert all(agent.glucose_detector is None for agent in disabled.citizen_agents)
