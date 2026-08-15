"""Tests for YAML/TOML configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from garland.app import build_config_from_args, parse_run_args
from garland.attacks import AttackType
from garland.config import (
    apply_overrides,
    config_from_dict,
    config_to_dict,
    load_config_file,
)
from garland.simulation import SimulationConfig


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


class TestCliConfigMerge:
    def test_config_file_with_cli_override(self, tmp_path: Path):
        path = tmp_path / "sim.yaml"
        path.write_text("n_agents: 400\nn_steps: 20\n", encoding="utf-8")
        args = parse_run_args(["--config", str(path), "--n-agents", "900"])
        config = build_config_from_args(args)
        assert config.n_agents == 900
        assert config.n_steps == 20

    def test_config_to_dict_roundtrip(self):
        original = SimulationConfig(n_agents=123, n_steps=7)
        restored = config_from_dict(config_to_dict(original))
        assert restored.n_agents == 123
        assert restored.n_steps == 7
        assert restored.anomaly_threshold == pytest.approx(original.anomaly_threshold)

    def test_unknown_attack_type_raises(self):
        with pytest.raises(ValueError, match="Unknown attack type"):
            config_from_dict({"attacks": {"active_attacks": ["not_real"]}})
