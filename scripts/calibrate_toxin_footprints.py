"""Calibrate Gaussian-plume release rates against physical footprint measures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from garland.hazards import (
    PlumeConfig,
    compute_plume_concentrations,
    concentration_for_respiratory_delta,
)
from garland.paths import resolve_under_base, write_json_file

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("output/toxin_footprint_calibration.json")
DEFAULT_RELEASE_RATES = (5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)
DEFAULT_STABILITY_CLASSES = ("A", "C", "D")


def _resolve_output_path(user_path: Path) -> Path:
    """Resolve an output argument beneath the repository output directory."""
    output_base = REPO / "output"
    relative_path = user_path
    if not user_path.is_absolute() and user_path.parts[:1] == ("output",):
        relative_path = Path(*user_path.parts[1:])
    return resolve_under_base(output_base, relative_path)


def calibrate_rate(
    release_rate: float,
    stability_class: str,
    *,
    grid_m: float = 2_000.0,
    sample_step_m: float = 5.0,
    wearable_density_per_km2: float = 375.0,
    minimum_respiratory_delta_bpm: float = 2.0,
) -> dict[str, float | bool | str]:
    """Measure one release-rate/stability pair on a regular concentration lattice."""
    if release_rate <= 0.0:
        raise ValueError("release_rate must be positive")
    if sample_step_m <= 0.0 or grid_m <= 0.0:
        raise ValueError("grid_m and sample_step_m must be positive")
    if wearable_density_per_km2 < 0.0:
        raise ValueError("wearable_density_per_km2 must be non-negative")

    gate = concentration_for_respiratory_delta(minimum_respiratory_delta_bpm)
    axis = np.arange(0.0, grid_m + sample_step_m * 0.5, sample_step_m)
    x_values, y_values = np.meshgrid(axis, axis, indexing="xy")
    source = grid_m / 2.0
    plume = PlumeConfig(
        plume_id=f"rate_{release_rate:g}_{stability_class}",
        source_x=source,
        source_y=source,
        release_rate=release_rate,
        wind_speed=1.0,
        wind_direction=0.0,
        stability_class=stability_class,
        start_step=864,
        duration_steps=288,
    )
    concentrations, _ = compute_plume_concentrations(
        x_values.ravel(),
        y_values.ravel(),
        [plume],
        900,
    )
    above_gate = concentrations.reshape(x_values.shape) > gate
    cell_area_m2 = sample_step_m * sample_step_m
    area_ha = float(np.count_nonzero(above_gate) * cell_area_m2 / 10_000.0)
    if np.any(above_gate):
        above_x = x_values[above_gate]
        above_y = y_values[above_gate]
        crosswind_span = float(np.max(above_y) - np.min(above_y))
        downwind_span = float(np.max(above_x) - np.min(above_x))
        downwind_clipped = bool(np.max(above_x) >= grid_m - sample_step_m * 0.5)
    else:
        crosswind_span = 0.0
        downwind_span = 0.0
        downwind_clipped = False
    implied_devices = area_ha * wearable_density_per_km2 / 100.0
    grid_wearable_population = wearable_density_per_km2 * grid_m * grid_m / 1_000_000.0
    return {
        "release_rate": release_rate,
        "stability_class": stability_class,
        "concentration_gate": gate,
        "footprint_area_hectares": area_ha,
        "crosswind_span_m": crosswind_span,
        "downwind_span_m": downwind_span,
        "downwind_span_clipped_by_grid": downwind_clipped,
        "wearable_density_per_km2": wearable_density_per_km2,
        "implied_device_count": implied_devices,
        "grid_wearable_population": grid_wearable_population,
    }


def calibrate(
    *,
    release_rates: tuple[float, ...] = DEFAULT_RELEASE_RATES,
    stability_classes: tuple[str, ...] = DEFAULT_STABILITY_CLASSES,
    grid_m: float = 2_000.0,
    sample_step_m: float = 5.0,
    wearable_density_per_km2: float = 375.0,
    minimum_respiratory_delta_bpm: float = 2.0,
) -> dict[str, object]:
    """Return the complete release-rate/stability calibration matrix."""
    rows = [
        calibrate_rate(
            release_rate,
            stability_class,
            grid_m=grid_m,
            sample_step_m=sample_step_m,
            wearable_density_per_km2=wearable_density_per_km2,
            minimum_respiratory_delta_bpm=minimum_respiratory_delta_bpm,
        )
        for stability_class in stability_classes
        for release_rate in release_rates
    ]
    return {
        "minimum_respiratory_delta_bpm": minimum_respiratory_delta_bpm,
        "concentration_gate": concentration_for_respiratory_delta(minimum_respiratory_delta_bpm),
        "grid_m": grid_m,
        "sample_step_m": sample_step_m,
        "wearable_density_per_km2": wearable_density_per_km2,
        "results": rows,
    }


def main() -> None:
    """Run the calibration matrix and write validated JSON output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = calibrate()
    output_path = _resolve_output_path(args.output)
    write_json_file(output_path, result)
    for row in result["results"]:
        print(
            f"{row['stability_class']} rate={row['release_rate']:g}: "
            f"area={row['footprint_area_hectares']:.3f} ha, "
            f"crosswind={row['crosswind_span_m']:.1f} m, "
            f"downwind={row['downwind_span_m']:.1f} m, "
            f"clipped={row['downwind_span_clipped_by_grid']}, "
            f"devices={row['implied_device_count']:.1f}"
        )


if __name__ == "__main__":
    main()
