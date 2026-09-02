"""Tests for biometric synthesis backends and Open Wearables export."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from garland.biometric_profiles import BiometricProfile, build_profile, generate_profiles
from garland.biometric_synthesis import (
    generate_observation,
    generate_observation_custom,
    generate_observation_neurokit,
    generate_observations_batch,
)
from garland.channels import ChannelSet
from garland.openwearables import export_timeseries_payload, observation_to_records


@pytest.fixture
def profile() -> BiometricProfile:
    return build_profile(
        resting={
            "heart_rate": 72.0,
            "hrv_rmssd": 42.0,
            "respiratory_rate": 15.0,
            "body_temperature": 36.8,
        },
        circadian_amp={
            "heart_rate": 5.0,
            "respiratory_rate": 1.0,
            "body_temperature": 0.3,
        },
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestCustomSynthesis:
    def test_observation_shape(self, profile, rng):
        obs = generate_observation_custom(profile, 12.0, 180, rng)
        assert obs.shape == (4,)

    def test_hrv_floor(self, profile, rng):
        obs = generate_observation_custom(profile, 12.0, 180, rng, activity_level=1.0)
        assert obs[1] >= 5.0

    def test_batch_matches_deterministic_per_agent_formula(self, profile):
        zero_noise_set = ChannelSet(
            tuple(replace(channel, noise_sd=0.0) for channel in profile.channel_set.channels)
        )
        profiles = [
            build_profile(
                resting={
                    channel.name: profile.resting[position]
                    for position, channel in enumerate(profile.channel_set.channels)
                },
                circadian_amp={
                    channel.name: profile.circadian_amp[position]
                    for position, channel in enumerate(profile.channel_set.channels)
                },
                channel_set=zero_noise_set,
            ),
            build_profile(channel_set=zero_noise_set),
        ]
        resting = np.array([item.resting for item in profiles])
        amplitudes = np.array([item.circadian_amp for item in profiles])
        activity = np.array([0.15, 0.0])
        batch = generate_observations_batch(
            resting,
            amplitudes,
            zero_noise_set,
            12.0,
            180,
            np.random.default_rng(42),
            activity,
        )
        expected = np.array(
            [
                generate_observation_custom(
                    item,
                    12.0,
                    180,
                    np.random.default_rng(100 + index),
                    activity_level=float(activity[index]),
                )
                for index, item in enumerate(profiles)
            ]
        )
        np.testing.assert_allclose(batch, expected)

    def test_batch_noise_distribution_matches_channel_parameters(self, profile):
        n = 10000
        resting = np.repeat(profile.resting[None, :], n, axis=0)
        amplitudes = np.repeat(profile.circadian_amp[None, :], n, axis=0)
        activity = np.zeros(n)
        observations = generate_observations_batch(
            resting,
            amplitudes,
            profile.channel_set,
            12.0,
            180,
            np.random.default_rng(43),
            activity,
        )
        zero_noise_set = ChannelSet(
            tuple(replace(channel, noise_sd=0.0) for channel in profile.channel_set.channels)
        )
        deterministic = generate_observations_batch(
            resting,
            amplitudes,
            zero_noise_set,
            12.0,
            180,
            np.random.default_rng(44),
            activity,
        )
        expected_sd = np.array([channel.noise_sd for channel in profile.channel_set.channels])
        assert np.all(np.isfinite(observations))
        np.testing.assert_allclose(observations.mean(axis=0), deterministic.mean(axis=0), atol=0.12)
        np.testing.assert_allclose(observations.std(axis=0), expected_sd, rtol=0.05, atol=0.05)

    def test_batch_respects_channel_floors(self, profile):
        resting = np.full((32, len(profile.channel_set)), -100.0)
        amplitudes = np.zeros_like(resting)
        observations = generate_observations_batch(
            resting,
            amplitudes,
            profile.channel_set,
            12.0,
            180,
            np.random.default_rng(45),
            np.full(32, -10.0),
        )
        floors = np.array(
            [
                channel.floor if channel.floor is not None else -np.inf
                for channel in profile.channel_set.channels
            ]
        )
        assert np.all(observations >= floors)


class TestNeurokitSynthesis:
    @pytest.fixture(autouse=True)
    def _require_neurokit(self):
        pytest.importorskip("neurokit2")

    def test_neurokit_observation_shape(self, profile, rng):
        obs = generate_observation_neurokit(profile, 12.0, 180, rng, window_seconds=30.0)
        assert obs.shape == (4,)

    def test_neurokit_backend_dispatch(self, profile):
        dispatched = generate_observation(
            profile,
            12.0,
            180,
            np.random.default_rng(46),
            backend="neurokit",
            neurokit_window_seconds=30.0,
        )
        direct = generate_observation_neurokit(
            profile,
            12.0,
            180,
            np.random.default_rng(46),
            window_seconds=30.0,
        )
        np.testing.assert_allclose(dispatched, direct)

    def test_neurokit_hr_tracks_profile(self, profile, rng):
        low = build_profile(resting={"heart_rate": 60.0})
        high = build_profile(resting={"heart_rate": 90.0})
        low_obs = generate_observation_neurokit(low, 12.0, 180, rng, window_seconds=30.0)
        high_obs = generate_observation_neurokit(high, 12.0, 180, rng, window_seconds=30.0)
        assert high_obs[0] > low_obs[0]

    def test_neurokit_matches_custom_statistics(self):
        """NeuroKit2 and custom backends should produce similar population stats."""
        profiles = generate_profiles(20, np.random.default_rng(0))
        custom_rng = np.random.default_rng(1)
        neurokit_rng = np.random.default_rng(1)
        custom = np.array([generate_observation_custom(p, 14.0, 180, custom_rng) for p in profiles])
        neurokit = np.array(
            [
                generate_observation_neurokit(p, 14.0, 180, neurokit_rng, window_seconds=60)
                for p in profiles
            ]
        )
        for dim, tol in [(0, 12.0), (2, 5.0), (3, 1.0)]:
            assert abs(custom[:, dim].mean() - neurokit[:, dim].mean()) < tol
        assert 5.0 <= neurokit[:, 1].mean() <= 120.0
        assert 5.0 <= custom[:, 1].mean() <= 120.0

    def test_missing_neurokit_raises(self, profile, rng, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "neurokit2":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(ImportError, match="biosignals"):
            generate_observation_neurokit(profile, 12.0, 180, rng)


class TestOpenWearablesExport:
    def test_record_fields(self, profile, rng):
        obs = generate_observation_custom(profile, 12.0, 180, rng)
        ts = datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc)
        records = observation_to_records(obs, ts)
        assert len(records) == 4
        types = {r["type"] for r in records}
        assert types == {
            "heart_rate",
            "heart_rate_variability_rmssd",
            "respiratory_rate",
            "body_temperature",
        }
        for record in records:
            assert record["timestamp"] == "2024-06-15T14:30:00Z"
            assert record["zone_offset"] == "+00:00"
            assert record["source"] == "garland"
            assert isinstance(record["value"], float)
            assert record["unit"]

    def test_timeseries_payload_shape(self, profile, rng):
        obs = generate_observation_custom(profile, 12.0, 180, rng)
        records = observation_to_records(obs, datetime(2024, 1, 1, tzinfo=timezone.utc))
        payload = export_timeseries_payload(records, resolution="5min")
        assert "data" in payload
        assert "pagination" in payload
        assert "metadata" in payload
        assert payload["metadata"]["resolution"] == "5min"
        assert payload["metadata"]["sample_count"] == 4

    def test_invalid_observation_length_raises(self):
        observation = np.array([1.0, 2.0])
        timestamp = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="Expected 4-dimensional"):
            observation_to_records(observation, timestamp)


class TestSynthesisDispatcher:
    def test_custom_backend_default(self, profile):
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        obs = generate_observation(profile, 12.0, 180, rng_a, backend="custom")
        expected = generate_observation_custom(profile, 12.0, 180, rng_b)
        np.testing.assert_allclose(obs, expected)

    def test_unknown_backend_raises(self, profile, rng):
        with pytest.raises(ValueError, match="Unknown biometric synthesis"):
            generate_observation(profile, 12.0, 180, rng, backend="wavelet")  # type: ignore[arg-type]
