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

    def test_nested_sections(self):
        config = config_from_dict(
            {
                "n_agents": 500,
                "privacy": {"epsilon_per_response": 0.05, "k_min": 25},
                "attacks": {"active_attacks": ["sybil_injection", "replay"]},
            }
        )
        assert config.n_agents == 500
        assert config.privacy.epsilon_per_response == 0.05
        assert config.privacy.k_min == 25
        assert config.attacks.active_attacks == [AttackType.SYBIL_INJECTION, AttackType.REPLAY]

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
        assert merged["privacy"]["epsilon_per_response"] == 0.2


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

    def test_unknown_attack_type_raises(self):
        with pytest.raises(ValueError, match="Unknown attack type"):
            config_from_dict({"attacks": {"active_attacks": ["not_real"]}})
