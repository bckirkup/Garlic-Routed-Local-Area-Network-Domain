"""Tests for the GARLAND CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from garland.app import build_config_from_args, main, parse_args


class TestParseArgs:
    """CLI argument parsing."""

    def test_defaults(self):
        args = parse_args([])
        assert args.n_agents == 250_000
        assert args.n_steps == 2016
        assert args.wearable_fraction == 0.15
        assert args.enable_sybil is False
        assert args.enable_deanon is False
        assert args.enable_correlation is False
        assert args.enable_eclipse is False
        assert args.enable_replay is False

    def test_enable_sybil_flag(self):
        args = parse_args(["--enable-sybil", "--sybil-count", "5"])
        assert args.enable_sybil is True
        assert args.sybil_count == 5

    def test_enable_deanon_flag(self):
        args = parse_args(["--enable-deanon"])
        assert args.enable_deanon is True

    def test_enable_correlation_flag(self):
        args = parse_args(["--enable-correlation"])
        assert args.enable_correlation is True

    def test_enable_eclipse_flag(self):
        args = parse_args(["--enable-eclipse", "--eclipse-zones", "1,2,3"])
        assert args.enable_eclipse is True
        assert args.eclipse_zones == "1,2,3"

    def test_enable_replay_flag(self):
        args = parse_args(["--enable-replay"])
        assert args.enable_replay is True

    def test_attack_target_agent_flag(self):
        args = parse_args(["--attack-target-agent", "42"])
        assert args.attack_target_agent == 42

    def test_custom_output_dir(self, tmp_path: Path):
        args = parse_args(["--output-dir", str(tmp_path / "out")])
        assert args.output_dir == str(tmp_path / "out")

    def test_export_openwearables_flags(self):
        args = parse_args(
            [
                "--export-openwearables",
                "wearables.json",
                "--openwearables-max-agents",
                "5",
            ]
        )
        assert args.export_openwearables == "wearables.json"
        assert args.openwearables_max_agents == 5

    def test_config_flag(self, tmp_path: Path):
        config_path = tmp_path / "sim.yaml"
        config_path.write_text("n_agents: 500\nn_steps: 10\n", encoding="utf-8")
        args = parse_args(["--config", str(config_path)])
        assert args.config == str(config_path)


class TestBuildConfig:
    def test_config_file_applied(self, tmp_path: Path):
        config_path = tmp_path / "sim.yaml"
        config_path.write_text("n_agents: 600\nn_steps: 11\n", encoding="utf-8")
        args = parse_args(["--config", str(config_path)])
        config = build_config_from_args(args)
        assert config.n_agents == 600
        assert config.n_steps == 11


class TestMain:
    """End-to-end CLI smoke tests."""

    def test_quick_run_writes_outputs(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        main(
            [
                "--n-agents",
                "300",
                "--n-steps",
                "20",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )
        captured = capsys.readouterr()
        assert "Done." in captured.out

        csv_path = tmp_path / "simulation_metrics.csv"
        json_path = tmp_path / "summary.json"
        assert csv_path.exists()
        assert json_path.exists()

        summary = json.loads(json_path.read_text())
        assert "total_epsilon" in summary
        assert "fpr_disease" in summary

    def test_sybil_flag_runs(self, tmp_path: Path):
        main(
            [
                "--n-agents",
                "400",
                "--n-steps",
                "25",
                "--enable-sybil",
                "--sybil-count",
                "10",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["total_broadcasts"] >= 0

    def test_deanon_flag_runs(self, tmp_path: Path):
        main(
            [
                "--n-agents",
                "400",
                "--n-steps",
                "50",
                "--enable-deanon",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "deanon_attempts" in summary

    def test_correlation_flag_runs(self, tmp_path: Path):
        main(
            [
                "--n-agents",
                "400",
                "--n-steps",
                "30",
                "--enable-correlation",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "correlation_evaluations" in summary

    def test_eclipse_flag_runs(self, tmp_path: Path):
        main(
            [
                "--n-agents",
                "400",
                "--n-steps",
                "25",
                "--enable-eclipse",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "eclipse_tokens_dropped" in summary

    def test_replay_flag_runs(self, tmp_path: Path):
        main(
            [
                "--n-agents",
                "400",
                "--n-steps",
                "20",
                "--enable-sybil",
                "--enable-replay",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "replay_tokens_injected" in summary

    def test_config_file_run(self, tmp_path: Path):
        config_path = tmp_path / "sim.yaml"
        config_path.write_text(
            "n_agents: 300\nn_steps: 15\nprivacy:\n  k_min: 10\n",
            encoding="utf-8",
        )
        main(
            [
                "--config",
                str(config_path),
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "total_epsilon" in summary

    def test_export_openwearables_writes_json(self, tmp_path: Path):
        main(
            [
                "--n-agents",
                "300",
                "--n-steps",
                "10",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
                "--export-openwearables",
                "openwearables.json",
                "--openwearables-max-agents",
                "3",
            ]
        )
        export_path = tmp_path / "openwearables.json"
        assert export_path.exists()
        payload = json.loads(export_path.read_text())
        assert "data" in payload
        assert "metadata" in payload
        assert payload["metadata"]["resolution"] == "5min"
        assert payload["metadata"]["sample_count"] > 0
        assert payload["metadata"]["start_time"] is not None
        assert payload["metadata"]["end_time"] is not None
        for record in payload["data"]:
            assert "type" in record
            assert "value" in record
            assert "unit" in record
            assert "timestamp" in record

    def test_sweep_subcommand(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        base_config = tmp_path / "base.yaml"
        base_config.write_text("n_agents: 200\nn_steps: 8\n", encoding="utf-8")
        sweep_config = tmp_path / "sweep.yaml"
        sweep_config.write_text(
            "\n".join(
                [
                    f"base_config: {base_config}",
                    f"output_dir: {tmp_path / 'sweep_out'}",
                    "sweep:",
                    "  privacy.epsilon_per_response: [0.05, 0.1]",
                ]
            ),
            encoding="utf-8",
        )
        main(["sweep", "--sweep-config", str(sweep_config)])
        captured = capsys.readouterr()
        assert "Sweep complete" in captured.out
        assert (tmp_path / "sweep_out" / "sweep_results.csv").exists()
