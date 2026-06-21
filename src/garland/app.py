"""GARLAND CLI entry point.

Run the epidemiological security testbed simulation with configurable parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from garland.attacks import AttackConfig, AttackType  # noqa: F401
from garland.hazards import PlumeConfig, SEIRConfig
from garland.privacy import PrivacyConfig
from garland.simulation import GarlandModel, SimulationConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GARLAND: Privacy-Preserving Epidemiological Security Testbed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Population & scale
    parser.add_argument(
        "--n-agents", type=int, default=250_000, help="Total population size"
    )
    parser.add_argument(
        "--wearable-fraction",
        type=float,
        default=0.15,
        help="Fraction of agents with wearable devices",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2016,
        help="Simulation steps (each = 5 minutes; 2016 = 7 days)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Spatial
    parser.add_argument(
        "--grid-width", type=float, default=10_000.0, help="Domain width (meters)"
    )
    parser.add_argument(
        "--grid-height", type=float, default=10_000.0, help="Domain height (meters)"
    )
    parser.add_argument(
        "--cell-size", type=float, default=200.0, help="Grid cell size (meters)"
    )

    # Baseline / forgetting
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=0.01,
        help="Exponential decay rate for biometric baseline forgetting",
    )
    parser.add_argument(
        "--seasonal-decay",
        type=float,
        default=0.001,
        help="Seasonal pattern learning rate",
    )

    # SEIR
    parser.add_argument("--seir-beta", type=float, default=0.015, help="SEIR beta")
    parser.add_argument(
        "--seir-sigma", type=float, default=0.000694, help="SEIR sigma (E→I rate)"
    )
    parser.add_argument(
        "--seir-gamma", type=float, default=0.000347, help="SEIR gamma (I→R rate)"
    )
    parser.add_argument(
        "--initial-infected", type=int, default=10, help="Initial seed infections"
    )

    # Plume
    parser.add_argument(
        "--plume-start-step",
        type=int,
        default=288,
        help="Step when plume begins (288 = 24h)",
    )
    parser.add_argument(
        "--plume-duration", type=int, default=144, help="Plume duration (steps)"
    )
    parser.add_argument(
        "--plume-x", type=float, default=5000.0, help="Plume source X (meters)"
    )
    parser.add_argument(
        "--plume-y", type=float, default=5000.0, help="Plume source Y (meters)"
    )

    # Privacy
    parser.add_argument(
        "--threshold-m",
        type=int,
        default=5,
        help="Anomaly count threshold for broadcast trigger",
    )
    parser.add_argument(
        "--k-min", type=int, default=50, help="K-anonymity minimum population"
    )
    parser.add_argument(
        "--epsilon-per-response",
        type=float,
        default=0.1,
        help="Privacy budget per randomized response",
    )
    parser.add_argument(
        "--rr-probability",
        type=float,
        default=0.75,
        help="Randomized response truthful probability",
    )
    parser.add_argument(
        "--laplace-scale",
        type=float,
        default=200.0,
        help="Planar Laplace scale (meters)",
    )

    # Attacks
    parser.add_argument(
        "--enable-sybil", action="store_true", help="Enable Sybil injection attack"
    )
    parser.add_argument(
        "--enable-deanon",
        action="store_true",
        help="Enable deanonymization attack",
    )
    parser.add_argument(
        "--enable-correlation",
        action="store_true",
        help="Enable temporal/spatial correlation attack",
    )
    parser.add_argument(
        "--enable-eclipse",
        action="store_true",
        help="Enable eclipse (token interception) attack",
    )
    parser.add_argument(
        "--enable-replay",
        action="store_true",
        help="Enable stale token replay attack",
    )
    parser.add_argument(
        "--sybil-count", type=int, default=20, help="Sybil identities per injection"
    )
    parser.add_argument(
        "--sybil-target-zone",
        type=int,
        default=0,
        help="Grid cell ID for Sybil flooding (0 = target agent cell)",
    )
    parser.add_argument(
        "--attack-target-agent",
        type=int,
        default=0,
        help="Agent index targeted by deanon/correlation attacks",
    )
    parser.add_argument(
        "--eclipse-zones",
        type=str,
        default="",
        help="Comma-separated grid cell IDs to eclipse (empty = target agent cell)",
    )
    parser.add_argument(
        "--correlation-window",
        type=int,
        default=288,
        help="Observation window (steps) for correlation attack",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for CSV and plot outputs",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip plot generation"
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the GARLAND simulation."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build attack config
    active_attacks: list[AttackType] = []
    if args.enable_sybil:
        active_attacks.append(AttackType.SYBIL_INJECTION)
    if args.enable_deanon:
        active_attacks.append(AttackType.TARGETED_QUERY)
    if args.enable_correlation:
        active_attacks.append(AttackType.CORRELATION)
    if args.enable_eclipse:
        active_attacks.append(AttackType.ECLIPSE)
    if args.enable_replay:
        active_attacks.append(AttackType.REPLAY)

    eclipse_zones: list[int] = []
    if args.eclipse_zones.strip():
        eclipse_zones = [int(z.strip()) for z in args.eclipse_zones.split(",") if z.strip()]

    config = SimulationConfig(
        n_agents=args.n_agents,
        wearable_fraction=args.wearable_fraction,
        n_steps=args.n_steps,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        cell_size=args.cell_size,
        seed=args.seed,
        baseline_decay_lambda=args.decay_lambda,
        baseline_seasonal_decay=args.seasonal_decay,
        seir=SEIRConfig(
            beta=args.seir_beta,
            sigma=args.seir_sigma,
            gamma=args.seir_gamma,
            initial_infected=args.initial_infected,
        ),
        plume=PlumeConfig(
            source_x=args.plume_x,
            source_y=args.plume_y,
            start_step=args.plume_start_step,
            duration_steps=args.plume_duration,
        ),
        privacy=PrivacyConfig(
            threshold_m=args.threshold_m,
            k_min=args.k_min,
            epsilon_per_response=args.epsilon_per_response,
            randomized_response_p=args.rr_probability,
            laplace_scale=args.laplace_scale,
        ),
        attacks=AttackConfig(
            sybil_count=args.sybil_count,
            sybil_target_zone=args.sybil_target_zone,
            target_agent_idx=args.attack_target_agent,
            correlation_window=args.correlation_window,
            eclipse_target_zones=eclipse_zones,
            active_attacks=active_attacks,
        ),
    )

    print("GARLAND Epidemiological Security Testbed")
    print("=" * 50)
    print(f"Population: {config.n_agents:,} agents")
    print(f"Wearable penetration: {config.wearable_fraction*100:.1f}%")
    print(f"Simulation: {config.n_steps} steps ({config.n_steps * 5 / 60:.1f} hours)")
    print(f"Spatial: {config.grid_width:.0f}m × {config.grid_height:.0f}m grid")
    print(f"Privacy: ε={config.privacy.epsilon_per_response}/response, K={config.privacy.k_min}")
    print(f"Attacks: {[a.value for a in active_attacks] if active_attacks else 'None'}")
    print(f"Output: {output_dir}")
    print()

    # Run simulation
    print("Initializing model...")
    model = GarlandModel(config)
    print(
        f"  Wearable agents: {int(sum(model.has_wearable)):,} "
        f"({int(sum(model.has_wearable))/config.n_agents*100:.1f}%)"
    )
    print(f"  Spatial cells: {model.grid.n_cells}")
    print()

    print("Running simulation...")
    for step_idx in range(config.n_steps):
        model.step()
        if (step_idx + 1) % 288 == 0:  # Report every 24h
            hours = (step_idx + 1) * 5 / 60
            seir_i = int(np.sum(model.seir.states == 2))
            print(
                f"  Day {int(hours/24)}: "
                f"Infectious={seir_i:,}, "
                f"ε={model.aggregator.state.total_epsilon:.3f}, "
                f"Broadcasts={model.aggregator.broadcasts_issued}"
            )

    # Output results
    print()
    print("Results")
    print("-" * 50)
    summary = model.metrics.summary()
    for key, value in summary.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Export CSV
    csv_path = output_dir / "simulation_metrics.csv"
    model.metrics.export_csv(csv_path)
    print(f"\nMetrics CSV: {csv_path}")

    # Export summary JSON
    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary JSON: {json_path}")

    # Generate plots
    if not args.no_plots:
        try:
            model.metrics.plot_metrics(output_dir)
            print(f"Plots: {output_dir}/*.png")
        except ImportError:
            print("Warning: matplotlib not available, skipping plots")

    print("\nDone.")


if __name__ == "__main__":
    main()
