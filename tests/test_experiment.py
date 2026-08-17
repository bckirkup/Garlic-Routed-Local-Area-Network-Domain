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
        marker_columns = {
            "world_settling_steps",
            "world_settling_complete",
            "world_settling_status",
            "steps_before_world_settling",
            "steps_after_world_settling",
            "world_settling_fraction_of_run",
            "fleet_cold_start",
            "fleet_cold_baseline_wearable_step_fraction",
            "post_world_settling_cold_baseline_wearable_step_fraction",
            "device_re_adoption_count",
            "legacy_device_adoption_warmup_reset_count",
        }
        assert marker_columns <= set(results.columns)
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

    def test_detection_power_columns_grade_with_adoption(self, tmp_path: Path):
        """Wider fleets must show up as wider scored vectors in the sweep table."""
        sweep_config = tmp_path / "adoption.yaml"
        sweep_config.write_text(
            "\n".join(
                [
                    f"output_dir: {tmp_path / 'adoption_out'}",
                    "n_agents: 400",
                    "n_steps: 24",
                    "wearable_fraction: 0.5",
                    "runs:",
                    "  - name: core_only",
                    "    devices:",
                    "      enabled: false",
                    "  - name: one_band",
                    "    devices:",
                    "      enabled: true",
                    "      adoption:",
                    "        motion_actigraphy: 1.0",
                    "  - name: two_bands",
                    "    devices:",
                    "      enabled: true",
                    "      adoption:",
                    "        motion_actigraphy: 1.0",
                    "        respiratory_acoustic_patch: 1.0",
                ]
            ),
            encoding="utf-8",
        )

        results = run_sweep(sweep_config).set_index("run_name")
        assert {"dp_scored_epochs", "dp_mean_effective_width"} <= set(results.columns)
        assert {
            "dp_width_1_5_true_positive_rate",
            "dp_width_25plus_false_positive_rate",
        } <= set(results.columns)

        widths = [results.loc[name, "dp_mean_effective_width"] for name in results.index]
        assert all(width is not None for width in widths)
        assert min(widths) > 0
        assert widths[0] < widths[1]
        assert widths[1] < widths[2]

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
