"""Experiment runner for parameter sweeps over GARLAND simulations."""

from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from garland.config import (
    _load_mapping,
    apply_overrides,
    config_from_dict,
    config_to_dict,
    load_config_file,
)
from garland.paths import ensure_directory, resolve_under_base, resolve_user_path, write_json_file
from garland.simulation import GarlandModel, SimulationConfig

_SUMMARY_COLUMNS = [
    "run_id",
    "run_name",
    "total_epsilon",
    "time_to_detection_disease_steps",
    "time_to_detection_toxin_steps",
    "fpr_disease",
    "fnr_disease",
    "fpr_toxin",
    "fnr_toxin",
    "discrimination_score",
    "total_broadcasts",
    "total_responses",
]


def run_simulation(
    config: SimulationConfig,
    *,
    write_outputs: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one simulation and return its summary metrics."""
    model = GarlandModel(config)
    metrics = model.run(config.n_steps)
    summary = metrics.summary()

    if write_outputs and output_dir is not None:
        safe_output_dir = ensure_directory(output_dir)
        metrics.export_csv(safe_output_dir / "simulation_metrics.csv")
        write_json_file(
            resolve_under_base(safe_output_dir, "summary.json"),
            summary,
        )

    return summary


def _expand_sweep_axes(sweep_axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not sweep_axes:
        raise ValueError("Sweep config must define at least one parameter axis")

    keys = list(sweep_axes.keys())
    values = [list(sweep_axes[key]) for key in keys]
    combinations: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo, strict=True)))
    return combinations


def _resolve_run_specs(sweep_data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reserved = {"runs", "sweep", "output_dir", "base_config", "base"}
    base = deepcopy(sweep_data.get("base", {}))
    for key, value in sweep_data.items():
        if key not in reserved:
            base[key] = deepcopy(value)

    if "base_config" in sweep_data:
        base_path = resolve_user_path(sweep_data["base_config"])
        base = apply_overrides(config_to_dict(load_config_file(base_path)), base)

    runs = sweep_data.get("runs")
    if runs is not None:
        if not isinstance(runs, list) or not runs:
            raise ValueError("Sweep config 'runs' must be a non-empty list")
        return base, list(runs)

    sweep_axes = sweep_data.get("sweep")
    if sweep_axes is None:
        raise ValueError("Sweep config must define either 'runs' or 'sweep'")

    if not isinstance(sweep_axes, dict):
        raise ValueError("Sweep config 'sweep' must be a mapping of parameter paths to values")

    run_specs: list[dict[str, Any]] = []
    for index, overrides in enumerate(_expand_sweep_axes(sweep_axes)):
        run_specs.append({"name": f"run_{index:03d}", **overrides})
    return base, run_specs


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    """Load a sweep definition from YAML or TOML."""
    return _load_mapping(resolve_user_path(path))


def run_sweep(
    sweep_config: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path | None = None,
    write_run_outputs: bool = False,
) -> pd.DataFrame:
    """Execute a parameter sweep and return a comparison table."""
    if not isinstance(sweep_config, dict):
        sweep_data = load_sweep_config(sweep_config)
    else:
        sweep_data = sweep_config

    base_overrides, run_specs = _resolve_run_specs(sweep_data)
    default_output = sweep_data.get("output_dir", "output/sweep")
    resolved_output_dir = ensure_directory(output_dir or default_output)

    rows: list[dict[str, Any]] = []
    for index, run_spec in enumerate(run_specs):
        if not isinstance(run_spec, dict):
            raise ValueError(f"Run spec at index {index} must be a mapping")

        run_spec = dict(run_spec)
        run_name = str(run_spec.pop("name", f"run_{index:03d}"))
        run_id = str(run_spec.pop("run_id", run_name))
        overrides = apply_overrides(base_overrides, run_spec)
        config = config_from_dict(overrides)

        run_output_dir = resolved_output_dir / run_id if write_run_outputs else None
        summary = run_simulation(
            config,
            write_outputs=write_run_outputs,
            output_dir=run_output_dir,
        )

        row: dict[str, Any] = {
            "run_id": run_id,
            "run_name": run_name,
        }
        for key, value in run_spec.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    row[f"param_{key}_{nested_key}"] = nested_value
            else:
                row[f"param_{key.replace('.', '_')}"] = value
        for column in _SUMMARY_COLUMNS[2:]:
            row[column] = summary.get(column)
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(resolved_output_dir / "sweep_results.csv", index=False)
    return results
