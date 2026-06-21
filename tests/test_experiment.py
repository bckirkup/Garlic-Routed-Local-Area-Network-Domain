"""Tests for the parameter sweep experiment runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from garland.experiment import run_sweep


class TestRunSweep:
    def test_grid_sweep_writes_comparison_csv(self, tmp_path: Path):
        sweep_config = tmp_path / "sweep.yaml"
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            "n_agents: 250\nn_steps: 12\nprivacy:\n  k_min: 10\n",
            encoding="utf-8",
        )
        sweep_config.write_text(
            "\n".join(
                [
                    f"base_config: {base_config}",
                    f"output_dir: {tmp_path / 'results'}",
                    "sweep:",
                    "  privacy.epsilon_per_response: [0.05, 0.1]",
                    "  privacy.k_min: [10, 20]",
                ]
            ),
            encoding="utf-8",
        )

        results = run_sweep(sweep_config)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 4
        assert "total_epsilon" in results.columns
        assert "fpr_disease" in results.columns
        assert (tmp_path / "results" / "sweep_results.csv").exists()

    def test_explicit_runs(self, tmp_path: Path):
        sweep_config = tmp_path / "runs.yaml"
        sweep_config.write_text(
            "\n".join(
                [
                    f"output_dir: {tmp_path / 'runs_out'}",
                    "n_agents: 200",
                    "n_steps: 10",
                    "runs:",
                    "  - name: low_epsilon",
                    "    privacy:",
                    "      epsilon_per_response: 0.05",
                    "  - name: high_epsilon",
                    "    privacy:",
                    "      epsilon_per_response: 0.2",
                ]
            ),
            encoding="utf-8",
        )

        results = run_sweep(sweep_config)
        assert len(results) == 2
        assert set(results["run_name"]) == {"low_epsilon", "high_epsilon"}

    def test_example_privacy_sweep(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "examples").mkdir()
        (tmp_path / "examples" / "quick.yaml").write_text(
            "n_agents: 250\nn_steps: 10\nprivacy:\n  k_min: 10\n",
            encoding="utf-8",
        )
        sweep_path = tmp_path / "examples" / "privacy_sweep.yaml"
        sweep_path.write_text(
            "\n".join(
                [
                    "base_config: examples/quick.yaml",
                    "output_dir: output/privacy_sweep",
                    "sweep:",
                    "  privacy.epsilon_per_response: [0.05, 0.1]",
                    "  privacy.k_min: [10]",
                ]
            ),
            encoding="utf-8",
        )

        results = run_sweep(sweep_path)
        assert len(results) == 2
