"""Tests for the GARLAND CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from garland.app import main, parse_args


class TestParseArgs:
    """CLI argument parsing."""

    def test_defaults(self):
        args = parse_args([])
        assert args.n_agents == 250_000
        assert args.n_steps == 2016
        assert args.wearable_fraction == 0.15
        assert args.enable_sybil is False
        assert args.enable_deanon is False

    def test_enable_sybil_flag(self):
        args = parse_args(["--enable-sybil", "--sybil-count", "5"])
        assert args.enable_sybil is True
        assert args.sybil_count == 5

    def test_enable_deanon_flag(self):
        args = parse_args(["--enable-deanon"])
        assert args.enable_deanon is True

    def test_custom_output_dir(self, tmp_path: Path):
        args = parse_args(["--output-dir", str(tmp_path / "out")])
        assert args.output_dir == str(tmp_path / "out")


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
