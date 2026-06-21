"""Optional performance benchmark for GARLAND at city scale.

Measures model initialization time, per-step throughput, and peak memory.
Use this to validate scaling assumptions before a full 7-day run.

Example::

    python -m garland.benchmark --n-agents 250000 --n-steps 20
    python -m garland.benchmark --quick
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

from garland.hazards import PlumeConfig
from garland.simulation import GarlandModel, SimulationConfig


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
        plume=PlumeConfig(start_step=10_000),
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        args.n_agents = 5_000
        args.n_steps = 10

    print("GARLAND scaling benchmark")
    print("=" * 50)
    print(f"Population: {args.n_agents:,} agents")
    print(f"Steps: {args.n_steps}")
    print()

    result = run_benchmark(args.n_agents, args.n_steps, args.seed)

    print(f"Wearable agents: {result['n_wearable']:,}")
    print(f"Init time: {result['init_seconds']:.2f}s")
    print(f"Peak init memory: {result['peak_init_mb']:.1f} MB")
    print(f"Avg step time: {result['avg_step_ms']:.1f} ms")
    print(f"Max step time: {result['max_step_ms']:.1f} ms")
    print(f"Peak step memory: {result['peak_step_mb']:.1f} MB")
    print(f"7-day extrapolation (avg step): {result['extrap_7d_hours']:.1f} hours")
    print()
    print(
        "Note: step time rises when hazards trigger more broadcasts and "
        "anomaly responses. See docs/SCALING.md for details."
    )


if __name__ == "__main__":
    main()
