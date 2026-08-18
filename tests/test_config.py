"""Tests for YAML/TOML configuration loading."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from garland.app import build_config_from_args, parse_run_args
from garland.attacks import AttackType
from garland.config import (
    apply_overrides,
    config_from_dict,
    config_to_dict,
    load_config_file,
)
from garland.simulation import GarlandModel, SimulationConfig
from garland.venues import VenueType


class TestConfigFromDict:
    def test_defaults(self):
        config = config_from_dict({})
        assert config.n_agents == 250_000
        assert config.privacy.k_min == 50
        assert config.spatial_backend == "hex"
        assert config.mobility_model == "random_walk"
        assert config.biometric_synthesis == "custom"
        assert config.anomaly_threshold == pytest.approx(3.5)
        assert config.detector_mode == "instant"
        assert config.privacy.dilation_basis == "observed_devices"
        assert config.privacy.dilation_window_steps == config.privacy.time_window_steps
        assert config.privacy.dilation_margin_factor == pytest.approx(0.5)

    def test_dilation_settings_validate_and_round_trip(self):
        config = config_from_dict(
            {
                "privacy": {
                    "dilation_basis": "residents",
                    "dilation_window_steps": 144,
                    "dilation_margin_factor": 1.5,
                    "enforce_release_k_anonymity": True,
                }
            }
        )
        restored = config_from_dict(config_to_dict(config))
        assert restored.privacy.dilation_basis == "residents"
        assert restored.privacy.dilation_window_steps == 144
        assert restored.privacy.dilation_margin_factor == pytest.approx(1.5)
        assert restored.privacy.enforce_release_k_anonymity is True

    def test_content_mechanism_settings_round_trip(self):
        config = config_from_dict(
            {
                "privacy": {
                    "response_mechanism": "aggregate_noisy_count",
                    "aggregate_count_epsilon": 0.4,
                    "aggregate_count_false_release_rate": 0.1,
                }
            }
        )
        restored = config_from_dict(config_to_dict(config))
        assert restored.privacy.response_mechanism == "aggregate_noisy_count"
        assert restored.privacy.aggregate_count_epsilon == pytest.approx(0.4)
        assert restored.privacy.aggregate_count_false_release_rate == pytest.approx(0.1)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dilation_basis", "invalid"),
            ("dilation_window_steps", 0),
            ("dilation_margin_factor", -1.0),
            ("response_mechanism", "invalid"),
            ("aggregate_count_epsilon", 0.0),
            ("aggregate_count_false_release_rate", 1.0),
        ],
    )
    def test_dilation_settings_reject_invalid_values(self, field, value):
        with pytest.raises(ValueError):
            config_from_dict({"privacy": {field: value}})

    def test_confounder_venue_types_round_trip_as_valid_tuple(self):
        config = config_from_dict(
            {
                "confounders": {
                    "venue_crowding_venue_types": [
                        VenueType.GATHERING.value,
                        VenueType.SPORTING.value,
                    ]
                }
            }
        )
        serialized = config_to_dict(config)
        assert serialized["confounders"]["venue_crowding_venue_types"] == [
            VenueType.GATHERING.value,
            VenueType.SPORTING.value,
        ]
        restored = config_from_dict(serialized)
        assert restored.confounders.venue_crowding_venue_types == (
            VenueType.GATHERING.value,
            VenueType.SPORTING.value,
        )

    def test_nested_sections(self):
        config = config_from_dict(
            {
                "n_agents": 500,
                "privacy": {"epsilon_per_response": 0.05, "k_min": 25},
                "attacks": {"active_attacks": ["sybil_injection", "replay"]},
            }
        )
        assert config.n_agents == 500
        assert config.privacy.epsilon_per_response == pytest.approx(0.05)
        assert config.privacy.k_min == 25
        assert config.attacks.active_attacks == [AttackType.SYBIL_INJECTION, AttackType.REPLAY]

    def test_confounder_section(self):
        config = config_from_dict(
            {
                "confounders": {
                    "enabled": True,
                    "exercise_rate": 0.2,
                    "heat_wave_duration_steps": 24,
                    "heat_wave_amplitude_jitter": 0.2,
                }
            }
        )
        assert config.confounders.enabled
        assert config.confounders.exercise_rate == pytest.approx(0.2)
        assert config.confounders.heat_wave_duration_steps == 24
        assert config.confounders.heat_wave_amplitude_jitter == pytest.approx(0.2)

    def test_exposure_and_sleep_jitter_round_trip(self):
        config = config_from_dict(
            {
                "confounders": {
                    "enabled": True,
                    "elderly_fraction": 0.3,
                    "has_air_conditioning_fraction": 0.6,
                    "outdoor_worker_fraction": 0.2,
                    "endurance_athlete_fraction": 0.15,
                    "heat_island_gain": 0.5,
                    "heat_wave_peak_hour": 16.0,
                    "heat_wave_materiality_floor": 0.6,
                    "sleep_disruption_delay_jitter_steps": 18,
                    "block_fire_duration_steps": 7,
                    "block_fire_center_x": 123.0,
                    "block_fire_radius_m": 42.0,
                    "block_fire_materiality_floor": 0.35,
                    "victory_duration_steps": 9,
                    "victory_fan_fraction": 0.4,
                    "victory_participation_fraction": 0.7,
                    "victory_onset_jitter_steps": 4,
                }
            }
        )
        restored = config_from_dict(config_to_dict(config))
        assert restored.confounders.elderly_fraction == pytest.approx(0.3)
        assert restored.confounders.has_air_conditioning_fraction == pytest.approx(0.6)
        assert restored.confounders.heat_island_gain == pytest.approx(0.5)
        assert restored.confounders.heat_wave_peak_hour == pytest.approx(16.0)
        assert restored.confounders.heat_wave_materiality_floor == pytest.approx(0.6)
        assert restored.confounders.sleep_disruption_delay_jitter_steps == 18
        assert restored.confounders.block_fire_duration_steps == 7
        assert restored.confounders.block_fire_center_x == pytest.approx(123.0)
        assert restored.confounders.block_fire_radius_m == pytest.approx(42.0)
        assert restored.confounders.block_fire_materiality_floor == pytest.approx(0.35)
        assert restored.confounders.victory_duration_steps == 9
        assert restored.confounders.victory_fan_fraction == pytest.approx(0.4)
        assert restored.confounders.victory_participation_fraction == pytest.approx(0.7)
        assert restored.confounders.victory_onset_jitter_steps == 4

    def test_anomaly_threshold_and_baseline_parameters(self):
        config = config_from_dict(
            {
                "anomaly_threshold": 5.0,
                "baseline_decay_lambda": 0.02,
                "baseline_seasonal_decay": 0.003,
            }
        )
        assert config.anomaly_threshold == pytest.approx(5.0)
        assert config.baseline_decay_lambda == pytest.approx(0.02)
        assert config.baseline_seasonal_decay == pytest.approx(0.003)

    def test_baseline_maturation_round_trip(self):
        config = config_from_dict(
            {
                "baseline_maturation": {
                    "minimum_history_days": 7,
                    "maximum_history_days": 90,
                    "cadence_steps": 12,
                }
            }
        )
        restored = config_from_dict(config_to_dict(config))
        assert restored.baseline_maturation.minimum_history_days == 7
        assert restored.baseline_maturation.maximum_history_days == 90
        assert restored.baseline_maturation.cadence_steps == 12

    @pytest.mark.parametrize(
        "settings",
        [
            {"minimum_history_days": -1},
            {"minimum_history_days": 8, "maximum_history_days": 7},
            {"cadence_steps": 0},
            {"cadence_steps": 1.5},
        ],
    )
    def test_baseline_maturation_rejects_invalid_values(self, settings):
        with pytest.raises(ValueError):
            config_from_dict({"baseline_maturation": settings})

    def test_world_settling_accepts_deprecated_alias(self):
        config = config_from_dict({"background_burn_in_steps": 17})
        assert config.world_settling_steps == 17
        assert "background_burn_in_steps" not in config_to_dict(config)

    def test_sequential_detector_parameters(self):
        config = config_from_dict(
            {
                "detector_mode": "sequential",
                "sequential_reference_value": 1.8,
                "sequential_threshold": 12.0,
                "sequential_clear_steps": 4,
                "sequential_clear_fraction": 0.4,
                "sequential_residual_ewma_alpha": 0.3,
            }
        )
        assert config.detector_mode == "sequential"
        assert config.sequential_reference_value == pytest.approx(1.8)
        assert config.sequential_threshold == pytest.approx(12.0)
        assert config.sequential_clear_steps == 4
        assert config.sequential_clear_fraction == pytest.approx(0.4)
        assert config.sequential_residual_ewma_alpha == pytest.approx(0.3)

    def test_attack_enable_flags(self):
        config = config_from_dict({"attacks": {"enable_sybil": True, "enable_replay": True}})
        assert AttackType.SYBIL_INJECTION in config.attacks.active_attacks
        assert AttackType.REPLAY in config.attacks.active_attacks


class TestApplyOverrides:
    def test_dotted_paths(self):
        merged = apply_overrides(
            {"privacy": {"k_min": 50, "epsilon_per_response": 0.1}},
            {"privacy.epsilon_per_response": 0.2},
        )
        assert merged["privacy"]["k_min"] == 50
        assert merged["privacy"]["epsilon_per_response"] == pytest.approx(0.2)


class TestLoadConfigFile:
    def test_load_yaml(self, tmp_path: Path):
        path = tmp_path / "sim.yaml"
        path.write_text(
            "n_agents: 400\nn_steps: 15\nprivacy:\n  k_min: 12\n",
            encoding="utf-8",
        )
        config = load_config_file(path)
        assert config.n_agents == 400
        assert config.n_steps == 15
        assert config.privacy.k_min == 12

    def test_load_toml(self, tmp_path: Path):
        path = tmp_path / "sim.toml"
        path.write_text(
            "n_agents = 450\nn_steps = 18\n[privacy]\nk_min = 15\n",
            encoding="utf-8",
        )
        config = load_config_file(path)
        assert config.n_agents == 450
        assert config.n_steps == 18
        assert config.privacy.k_min == 15

    def test_example_quick_yaml(self):
        config = load_config_file("examples/quick.yaml")
        assert config.n_agents == 300
        assert config.plume.start_step == 10

    def test_simulation_examples_load_and_round_trip(self):
        examples_dir = Path("examples")
        example_paths = sorted(
            path
            for suffix in ("*.yaml", "*.toml")
            for path in examples_dir.glob(suffix)
            if not path.name.endswith("_sweep.yaml")
        )

        assert example_paths
        for path in example_paths:
            config = load_config_file(path)
            restored = config_from_dict(config_to_dict(config))
            assert config_to_dict(restored) == config_to_dict(config)

    def test_town_archetypes_load_and_round_trip(self):
        for path in sorted(Path("examples").glob("town_*.yaml")):
            config = load_config_file(path)
            restored = config_from_dict(config_to_dict(config))
            assert config_to_dict(restored) == config_to_dict(config)
            assert config.mobility_model == "schedule"
            assert config.venues.enabled

    def test_town_archetype_axes_form_ordered_sweep(self):
        names = ("college", "tourist", "mill", "retirement", "exurb")
        configs = {name: load_config_file(f"examples/town_{name}.yaml") for name in names}
        density = [
            configs[name].n_agents
            / (configs[name].grid_width * configs[name].grid_height / 1_000_000.0)
            for name in names
        ]
        elderly = [configs[name].confounders.elderly_fraction for name in names]

        assert density[:4] == sorted(density[:4], reverse=True)
        assert density[3] > density[4]
        assert elderly[3] > elderly[1] > elderly[2] > elderly[4] > elderly[0]
        wearable = {name: configs[name].wearable_fraction for name in names}
        assert wearable["retirement"] == max(wearable.values())
        assert wearable["mill"] == min(wearable.values())
        assert density[0] - density[1] > 20.0
        assert density[1] - density[2] > 20.0
        assert elderly[1] - elderly[2] > 0.01

    def test_dense_and_sparse_town_smoke_runs_have_bounded_metrics(self):
        for name in ("college", "exurb"):
            config = load_config_file(f"examples/town_{name}.yaml")
            model = GarlandModel(replace(config, n_steps=4, world_settling_steps=0))
            model.run()
            summary = model.metrics.summary()

            assert model.current_step == 4
            assert summary["total_detection_events"] >= 0
            for key in (
                "background_rate",
                "artifact_detection_rate",
                "unexplained_detection_rate",
            ):
                assert 0.0 <= summary[key] <= 1.0
            for value in summary.values():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    assert np.isfinite(value)


class TestCliConfigMerge:
    def test_config_file_with_cli_override(self, tmp_path: Path):
        path = tmp_path / "sim.yaml"
        path.write_text("n_agents: 400\nn_steps: 20\n", encoding="utf-8")
        args = parse_run_args(["--config", str(path), "--n-agents", "900"])
        config = build_config_from_args(args)
        assert config.n_agents == 900
        assert config.n_steps == 20

    def test_zero_ablation_rate_overrides_configured_rate(self, tmp_path: Path):
        """A diagnostic a file switched on has to be switchable off from the CLI.

        Zero is the meaningful "off" value here, so it cannot double as the
        flag's unset sentinel.
        """
        path = tmp_path / "sim.yaml"
        path.write_text("detection_power:\n  channel_ablation_rate: 0.2\n", encoding="utf-8")

        configured = build_config_from_args(parse_run_args(["--config", str(path)]))
        disabled = build_config_from_args(
            parse_run_args(["--config", str(path), "--channel-ablation-rate", "0.0"])
        )

        assert configured.detection_power.channel_ablation_rate == pytest.approx(0.2)
        assert disabled.detection_power.channel_ablation_rate == pytest.approx(0.0)

    def test_config_to_dict_roundtrip(self):
        original = SimulationConfig(n_agents=123, n_steps=7)
        restored = config_from_dict(config_to_dict(original))
        assert restored.n_agents == 123
        assert restored.n_steps == 7
        assert restored.anomaly_threshold == pytest.approx(original.anomaly_threshold)

    def test_unknown_attack_type_raises(self):
        with pytest.raises(ValueError, match="Unknown attack type"):
            config_from_dict({"attacks": {"active_attacks": ["not_real"]}})
