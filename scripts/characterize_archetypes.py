"""Characterization harness for the five GARLAND town archetypes.

Measurement only: no repository code is modified. Each archetype is run with
its committed configuration (same seed, same step count, same spatial backend)
and reads first-class dilation measurements recorded once per issued broadcast.

Per archetype we record:
  * wearable device population and density (devices per km^2)
  * occupied-cell wearable occupancy (sampled each step)
  * dilation actually achieved: cells per broadcast, wearable population inside
    the dilated zone, and whether k_min was met
  * cold-baseline (onboarding / never-learned baseline) wearable-step fraction
  * explanation multiplicity: distinct active benign instances whose affected
    agents intersect the dilated zone, at the moment the broadcast is issued
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from garland.config import load_config_file
from garland.paths import resolve_under_base, write_json_file
from garland.simulation import GarlandModel

REPO = Path(__file__).resolve().parents[1]
ARCHETYPES = ["college", "tourist", "mill", "retirement", "exurb"]
DEFAULT_OUTPUT = Path("output/archetype_characterization.json")


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def characterize(name: str, n_steps: int) -> dict:
    config = load_config_file(REPO / f"examples/town_{name}.yaml")
    config = replace(config, n_steps=n_steps)
    model = GarlandModel(config)

    area_km2 = config.grid_width * config.grid_height / 1_000_000.0
    wearable_idx = [int(i) for i in np.nonzero(model.has_wearable)[0]]
    wearable_set = set(wearable_idx)

    cell_area_km2 = config.cell_size * config.cell_size / 1_000_000.0
    broadcasts: list[dict] = []
    occupancy_samples: list[int] = []
    occupied_cell_counts: list[int] = []

    inner_dilate = model.grid.dilated_zone

    def instrumented_dilate(
        center_cell: int,
        k_min: int,
        population_fn=None,
    ) -> list[int]:
        cells = inner_dilate(center_cell, k_min, population_fn)
        pop = 0
        wearables_in_zone: set[int] = set()
        for cell in cells:
            members = model.grid.agents_in_cell(cell)
            pop += len(members)
            wearables_in_zone.update(m for m in members if m in wearable_set)
        instances = 0
        causes: set[str] = set()
        for instance in model._confounder_step.benign_instances.values():
            affected = instance.current_agents
            if affected and (affected & wearables_in_zone):
                instances += 1
                causes.add(instance.cause.value)
        broadcasts.append(
            {
                "step": model.current_step,
                "cells": len(cells),
                "population": pop,
                "wearables": len(wearables_in_zone),
                "k_min": k_min,
                "met_k": len(wearables_in_zone) >= k_min,
                "instances": instances,
                "causes": sorted(causes),
            }
        )
        return cells

    model.grid.dilated_zone = instrumented_dilate  # type: ignore[method-assign]

    settling = config.world_settling_steps
    started = time.time()
    for _ in range(n_steps):
        model.step()
        if model.current_step > settling:
            cell_ids = np.asarray(model.grid.cell_ids)
            if len(cell_ids) and wearable_idx:
                worn = cell_ids[np.asarray(wearable_idx)]
                _, counts = np.unique(worn, return_counts=True)
                occupied_cell_counts.append(int(len(counts)))
                occupancy_samples.extend(int(c) for c in counts)
    elapsed = time.time() - started

    summary = model.metrics.summary()
    post = [b for b in broadcasts if b["step"] > settling]
    multiplicity = [b["instances"] for b in post]

    return {
        "archetype": name,
        "elapsed_s": round(elapsed, 1),
        "n_steps": n_steps,
        "world_settling_steps": settling,
        "n_agents": config.n_agents,
        "wearables": len(wearable_idx),
        "area_km2": area_km2,
        "agent_density_per_km2": config.n_agents / area_km2,
        "wearable_density_per_km2": len(wearable_idx) / area_km2,
        "k_min": model.aggregator.config.k_min,
        "occupied_wearable_cells_mean": (
            statistics.fmean(occupied_cell_counts) if occupied_cell_counts else None
        ),
        "wearables_per_occupied_cell_mean": (
            statistics.fmean(occupancy_samples) if occupancy_samples else None
        ),
        "wearables_per_occupied_cell_p90": _pct([float(v) for v in occupancy_samples], 0.9),
        "broadcasts_post_settling": len(post),
        "dilated_cells_mean": summary["dilated_cells_mean"],
        "dilated_cells_median": summary["dilated_cells_p50"],
        "dilated_cells_p90": summary["dilated_cells_p90"],
        "dilated_area_km2_mean": (
            summary["dilated_cells_mean"] * cell_area_km2
            if summary["dilated_cells_mean"] is not None
            else None
        ),
        "zone_wearables_mean": summary["true_respondent_population_mean"],
        "fraction_broadcasts_meeting_k": summary["fraction_true_respondents_meeting_k"],
        "explanation_multiplicity_mean": (
            statistics.fmean([float(m) for m in multiplicity]) if multiplicity else None
        ),
        "fraction_broadcasts_no_explanation": (
            statistics.fmean([1.0 if m == 0 else 0.0 for m in multiplicity])
            if multiplicity
            else None
        ),
        "fraction_broadcasts_multi_explanation": (
            statistics.fmean([1.0 if m >= 2 else 0.0 for m in multiplicity])
            if multiplicity
            else None
        ),
        "cold_baseline_wearable_step_fraction": summary[
            "post_world_settling_cold_baseline_wearable_step_fraction"
        ],
        "total_broadcasts": summary["total_broadcasts"],
        "broadcasts_per_occupied_zone_per_day": summary["broadcasts_per_occupied_zone_per_day"],
        "broadcasts_per_1000_agents_per_day": summary["broadcasts_per_1000_agents_per_day"],
        "fraction_occupied_zones_alarming": summary["fraction_occupied_zones_alarming"],
        "unexplained_detection_rate": summary["unexplained_detection_rate"],
        "explained_detections": summary["explained_detections"],
        "warranted_detections": summary["warranted_detections"],
        "total_detection_events": summary["total_detection_events"],
        "epsilon_per_agent_per_day": summary["epsilon_per_agent_per_day"],
        "confounder_agents_affected_by_cause": {
            k: v for k, v in summary["confounder_agents_affected_by_cause"].items()
        },
    }


def _resolve_output_path(user_path: Path) -> Path:
    """Resolve an output argument beneath the repository output directory."""
    output_base = REPO / "output"
    relative_path = user_path
    if not user_path.is_absolute() and user_path.parts[:1] == ("output",):
        relative_path = Path(*user_path.parts[1:])
    return resolve_under_base(output_base, relative_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path (default: output/archetype_characterization.json)",
    )
    parser.add_argument("--steps", type=int, default=1152, help="Number of simulation steps")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    results = []
    for name in ARCHETYPES:
        result = characterize(name, args.steps)
        results.append(result)
        print(json.dumps(result), flush=True)
    write_json_file(_resolve_output_path(args.output), results, default=str)


if __name__ == "__main__":
    main()
