"""Tests for the parameter sweep experiment runner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from garland.config import apply_overrides, config_from_dict
from garland.experiment import _resolve_run_specs, load_sweep_config, run_sweep


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

    def test_results_report_the_directory_written_to(self, tmp_path: Path):
        """Callers must be able to name the CSV they got, not guess the default."""
        sweep_config = tmp_path / "reported.yaml"
        sweep_config.write_text(
            "\n".join(
                [
                    f"output_dir: {tmp_path / 'reported_out'}",
                    "n_agents: 200",
                    "n_steps: 10",
                    "sweep:",
                    "  privacy.k_min: [10, 20]",
                ]
            ),
            encoding="utf-8",
        )

        results = run_sweep(sweep_config)
        assert Path(results.attrs["output_dir"]) == tmp_path / "reported_out"
        assert (Path(results.attrs["output_dir"]) / "sweep_results.csv").exists()

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

    def test_wearer_density_grades_the_zone_layer(self, tmp_path: Path):
        """The claim `detection_power_density_sweep.yaml` exists to test.

        Zone triggers need several devices in one cell within the window, so
        raising the share of residents carrying one at fixed population must
        raise both the scored epochs the fleet contributes and what the
        aggregation layer emits.
        The population ladder cannot show this: climbing `n_agents` on a fixed
        grid moves residents and wearers per zone together.
        """
        sweep_config = tmp_path / "density.yaml"
        sweep_config.write_text(
            "\n".join(
                [
                    f"output_dir: {tmp_path / 'density_out'}",
                    "n_agents: 600",
                    "n_steps: 48",
                    "privacy:",
                    "  k_min: 10",
                    "  threshold_m: 3",
                    "sweep:",
                    "  wearable_fraction: [0.1, 0.4, 0.9]",
                ]
            ),
            encoding="utf-8",
        )

        results = run_sweep(sweep_config)
        observed = results["dp_scored_epochs"].tolist()
        broadcasts = results["total_broadcasts"].tolist()

        assert observed[0] < observed[1] < observed[2]
        assert broadcasts[0] < broadcasts[2]
        assert all(count >= 0 for count in broadcasts)
        # Response epsilon is spent per broadcast, so the sparse arm cannot cost
        # more than the dense one.
        assert results["total_epsilon"].tolist()[0] <= results["total_epsilon"].tolist()[2]

    def test_committed_sweeps_resolve_to_loadable_runs(self):
        """Shipped sweep configs are excluded from the example round-trip test.

        Nothing else executes them cheaply, so a sweep whose `base_config` path
        or override key stopped resolving would only surface in a multi-minute
        run. Resolve each one's run specs and materialise the merged config.
        """
        sweep_paths = sorted(Path("examples").glob("*_sweep.yaml"))
        assert sweep_paths

        for path in sweep_paths:
            base, run_specs = _resolve_run_specs(load_sweep_config(path))
            assert run_specs, path
            for spec in run_specs:
                overrides = {k: v for k, v in spec.items() if k != "name"}
                config = config_from_dict(apply_overrides(deepcopy(base), overrides))
                assert config.n_agents > 0, path
                assert 0.0 <= config.wearable_fraction <= 1.0, path

    def test_universal_arms_sit_above_the_density_sweep(self):
        """`detection_power_universal_sweep.yaml` is the ceiling of the density arm.

        It only measures a capability ceiling if every arm is denser than the
        plausible-adoption sweep's top arm, and only isolates observed people
        from channels per person if it leaves subsystem adoption alone.
        """
        density = load_sweep_config("examples/detection_power_density_sweep.yaml")
        universal = load_sweep_config("examples/detection_power_universal_sweep.yaml")

        assert universal["base_config"] == density["base_config"]
        assert min(universal["sweep"]["wearable_fraction"]) > max(
            density["sweep"]["wearable_fraction"]
        )
        assert max(universal["sweep"]["wearable_fraction"]) <= 1.0
        assert "devices" not in universal
        assert "devices.adoption" not in universal["sweep"]

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
