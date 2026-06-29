"""Optional performance benchmark for GARLAND at city scale.

Measures model initialization time, per-step throughput, and peak memory.
Use this to validate scaling assumptions before a full 7-day run.

Example::

    python -m garland.benchmark --n-agents 250000 --n-steps 20
    python -m garland.benchmark --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from garland.hazards import PlumeConfig
from garland.paths import resolve_user_path, write_text_file
from garland.simulation import GarlandModel, SimulationConfig

# Documented in docs/SCALING.md and .github/workflows/benchmark.yml
QUICK_THRESHOLDS: dict[str, float] = {
    "max_init_seconds": 30.0,
    "max_avg_step_ms": 2_000.0,
    "max_peak_step_mb": 200.0,
}

FULL_THRESHOLDS: dict[str, float] = {
    "max_init_seconds": 180.0,
    "max_avg_step_ms": 15_000.0,
    "max_peak_step_mb": 2_000.0,
}


def run_benchmark(
    n_agents: int,
    n_steps: int = 10,
    seed: int = 42,
) -> dict[str, float | int]:
    """Run a short benchmark and return timing/memory metrics."""
    config = SimulationConfig(
        n_agents=n_agents,
        n_steps=n_steps,
        seed=seed,
        plumes=[PlumeConfig(start_step=10_000)],
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    model = GarlandModel(config)
    init_seconds = time.perf_counter() - t0
    _, peak_init_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    step_times: list[float] = []
    tracemalloc.start()
    for _ in range(n_steps):
        t0 = time.perf_counter()
        model.step()
        step_times.append(time.perf_counter() - t0)
    _, peak_step_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_step_ms = float(np.mean(step_times) * 1000)
    max_step_ms = float(np.max(step_times) * 1000)
    n_wearable = int(np.sum(model.has_wearable))

    return {
        "n_agents": n_agents,
        "n_wearable": n_wearable,
        "n_steps": n_steps,
        "init_seconds": init_seconds,
        "avg_step_ms": avg_step_ms,
        "max_step_ms": max_step_ms,
        "peak_init_mb": peak_init_bytes / (1024 * 1024),
        "peak_step_mb": peak_step_bytes / (1024 * 1024),
        "extrap_7d_hours": avg_step_ms * 2016 / 1000 / 3600,
    }


def threshold_violations(
    result: dict[str, float | int],
    thresholds: dict[str, float],
) -> list[str]:
    """Return human-readable messages for metrics outside documented bounds."""
    violations: list[str] = []
    checks = (
        ("init_seconds", "max_init_seconds", "Init time"),
        ("avg_step_ms", "max_avg_step_ms", "Avg step time"),
        ("peak_step_mb", "max_peak_step_mb", "Peak step memory"),
    )
    for metric_key, threshold_key, label in checks:
        value = float(result[metric_key])
        limit = thresholds[threshold_key]
        if value > limit:
            if metric_key.endswith("_mb"):
                violations.append(f"{label} {value:.1f} MB exceeds {limit:.1f} MB")
            elif metric_key.endswith("_ms"):
                violations.append(f"{label} {value:.1f} ms exceeds {limit:.1f} ms")
            else:
                violations.append(f"{label} {value:.2f}s exceeds {limit:.2f}s")
    return violations


def assert_within_thresholds(
    result: dict[str, float | int],
    thresholds: dict[str, float],
) -> None:
    """Exit with code 1 when benchmark metrics exceed documented bounds."""
    violations = threshold_violations(result, thresholds)
    if not violations:
        return
    for message in violations:
        print(f"THRESHOLD VIOLATION: {message}", file=sys.stderr)
    raise SystemExit(1)


def format_benchmark_report(result: dict[str, float | int]) -> str:
    """Format benchmark metrics for console or artifact output."""
    lines = [
        "GARLAND scaling benchmark",
        "=" * 50,
        f"Population: {int(result['n_agents']):,} agents",
        f"Steps: {int(result['n_steps'])}",
        "",
        f"Wearable agents: {int(result['n_wearable']):,}",
        f"Init time: {result['init_seconds']:.2f}s",
        f"Peak init memory: {result['peak_init_mb']:.1f} MB",
        f"Avg step time: {result['avg_step_ms']:.1f} ms",
        f"Max step time: {result['max_step_ms']:.1f} ms",
        f"Peak step memory: {result['peak_step_mb']:.1f} MB",
        f"7-day extrapolation (avg step): {result['extrap_7d_hours']:.1f} hours",
        "",
        "Note: step time rises when hazards trigger more broadcasts and "
        "anomaly responses. See docs/SCALING.md for details.",
    ]
    return "\n".join(lines)


def write_benchmark_output(
    result: dict[str, float | int],
    output_path: Path,
    *,
    thresholds: dict[str, float] | None = None,
) -> None:
    """Write benchmark metrics and optional threshold metadata to disk."""
    payload: dict[str, Any] = {"metrics": result}
    if thresholds is not None:
        payload["thresholds"] = thresholds
        payload["violations"] = threshold_violations(result, thresholds)
    write_text_file(
        output_path,
        format_benchmark_report(result) + "\n\n" + json.dumps(payload, indent=2),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GARLAND simulation performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=250_000,
        help="Population size to benchmark",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20,
        help="Number of simulation steps to time",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast smoke benchmark at 5,000 agents",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 when metrics exceed documented CI thresholds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write benchmark report and JSON metrics to this path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        args.n_agents = 5_000
        args.n_steps = 10

    result = run_benchmark(args.n_agents, args.n_steps, args.seed)
    thresholds = QUICK_THRESHOLDS if args.quick else FULL_THRESHOLDS

    print(format_benchmark_report(result))

    if args.output:
        output_path = resolve_user_path(args.output)
        write_benchmark_output(result, output_path, thresholds=thresholds)
        print(f"\nBenchmark output: {output_path}")

    if args.check:
        assert_within_thresholds(result, thresholds)


if __name__ == "__main__":
    main()
