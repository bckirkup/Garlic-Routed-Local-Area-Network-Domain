"""GARLAND CLI entry point.

Run the epidemiological security testbed simulation with configurable parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from garland.config import apply_overrides, config_from_dict, config_to_dict, load_config_file
from garland.experiment import run_sweep
from garland.simulation import GarlandModel, SimulationConfig


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Register simulation run arguments on a parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/TOML simulation config (CLI flags override file values)",
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
    parser.add_argument(
        "--spatial-backend",
        choices=["hex", "rect"],
        default="hex",
        help="Spatial index backend (hex=H3, rect=rectangular grid)",
    )
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=9,
        help="H3 resolution when using hex backend (9 ≈ 200 m cells)",
    )
    parser.add_argument(
        "--mobility-model",
        choices=["random_walk", "static"],
        default="random_walk",
        help="Agent mobility model",
    )
    parser.add_argument(
        "--mobility-speed",
        type=float,
        default=50.0,
        help="Random-walk max displacement per step (meters)",
    )
    parser.add_argument(
        "--static-agents",
        action="store_true",
        help="Disable agent mobility (alias for --mobility-model static)",
    )
    parser.add_argument(
        "--biometric-synthesis",
        choices=["custom", "neurokit"],
        default="custom",
        help="Biometric observation backend (neurokit requires pip install -e \".[biosignals]\")",
    )
    parser.add_argument(
        "--neurokit-window",
        type=float,
        default=60.0,
        help="ECG/RSP simulation window (seconds) for neurokit synthesis",
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
    parser.add_argument(
        "--enable-device-lifecycle",
        action="store_true",
        help="Enable wearable battery, removal, and power-off simulation",
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


def _cli_overrides_from_args(args: argparse.Namespace) -> dict:
    """Build config overrides for CLI flags that differ from parser defaults."""
    parser = _run_argument_parser()
    defaults = parser.parse_args([])
    overrides: dict = {}

    scalar_fields = {
        "n_agents": "n_agents",
        "wearable_fraction": "wearable_fraction",
        "n_steps": "n_steps",
        "seed": "seed",
        "grid_width": "grid_width",
        "grid_height": "grid_height",
        "cell_size": "cell_size",
        "spatial_backend": "spatial_backend",
        "h3_resolution": "h3_resolution",
        "mobility_speed": "mobility_speed_m",
        "biometric_synthesis": "biometric_synthesis",
        "neurokit_window": "neurokit_window_seconds",
        "decay_lambda": "baseline_decay_lambda",
        "seasonal_decay": "baseline_seasonal_decay",
    }
    for arg_name, config_key in scalar_fields.items():
        if getattr(args, arg_name) != getattr(defaults, arg_name):
            overrides[config_key] = getattr(args, arg_name)

    if args.mobility_model != defaults.mobility_model:
        overrides["mobility_model"] = args.mobility_model
    if args.static_agents:
        overrides["mobility_model"] = "static"

    seir_fields = {
        "seir_beta": "beta",
        "seir_sigma": "sigma",
        "seir_gamma": "gamma",
        "initial_infected": "initial_infected",
    }
    seir_overrides = {}
    for arg_name, field_name in seir_fields.items():
        if getattr(args, arg_name) != getattr(defaults, arg_name):
            seir_overrides[field_name] = getattr(args, arg_name)
    if seir_overrides:
        overrides["seir"] = seir_overrides

    plume_fields = {
        "plume_start_step": "start_step",
        "plume_duration": "duration_steps",
        "plume_x": "source_x",
        "plume_y": "source_y",
    }
    plume_overrides = {}
    for arg_name, field_name in plume_fields.items():
        if getattr(args, arg_name) != getattr(defaults, arg_name):
            plume_overrides[field_name] = getattr(args, arg_name)
    if plume_overrides:
        overrides["plume"] = plume_overrides

    privacy_fields = {
        "threshold_m": "threshold_m",
        "k_min": "k_min",
        "epsilon_per_response": "epsilon_per_response",
        "rr_probability": "randomized_response_p",
        "laplace_scale": "laplace_scale",
    }
    privacy_overrides = {}
    for arg_name, field_name in privacy_fields.items():
        if getattr(args, arg_name) != getattr(defaults, arg_name):
            privacy_overrides[field_name] = getattr(args, arg_name)
    if privacy_overrides:
        overrides["privacy"] = privacy_overrides

    attack_overrides: dict = {}
    attack_scalar_fields = {
        "sybil_count": "sybil_count",
        "sybil_target_zone": "sybil_target_zone",
        "attack_target_agent": "target_agent_idx",
        "correlation_window": "correlation_window",
    }
    for arg_name, field_name in attack_scalar_fields.items():
        if getattr(args, arg_name) != getattr(defaults, arg_name):
            attack_overrides[field_name] = getattr(args, arg_name)

    if args.eclipse_zones.strip() != defaults.eclipse_zones.strip():
        attack_overrides["eclipse_zones"] = args.eclipse_zones

    active_attacks = []
    if args.enable_sybil:
        active_attacks.append("sybil_injection")
    if args.enable_deanon:
        active_attacks.append("targeted_query")
    if args.enable_correlation:
        active_attacks.append("correlation")
    if args.enable_eclipse:
        active_attacks.append("eclipse")
    if args.enable_replay:
        active_attacks.append("replay")
    default_active = []
    if defaults.enable_sybil:
        default_active.append("sybil_injection")
    if defaults.enable_deanon:
        default_active.append("targeted_query")
    if defaults.enable_correlation:
        default_active.append("correlation")
    if defaults.enable_eclipse:
        default_active.append("eclipse")
    if defaults.enable_replay:
        default_active.append("replay")
    if active_attacks != default_active:
        attack_overrides["active_attacks"] = active_attacks

    if attack_overrides:
        overrides["attacks"] = attack_overrides

    if args.enable_device_lifecycle != defaults.enable_device_lifecycle:
        overrides["device_lifecycle"] = {"enabled": args.enable_device_lifecycle}

    return overrides


def build_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    """Build a simulation config from CLI args and optional config file."""
    if args.config:
        base = config_to_dict(load_config_file(args.config))
        merged = apply_overrides(base, _cli_overrides_from_args(args))
        return config_from_dict(merged)
    return config_from_dict(apply_overrides({}, _cli_overrides_from_args(args)))


def _run_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for a single simulation run."""
    parser = argparse.ArgumentParser(
        description="GARLAND: Privacy-Preserving Epidemiological Security Testbed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_run_arguments(parser)
    return parser


def parse_run_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a single simulation run."""
    return _run_argument_parser().parse_args(argv)


def parse_sweep_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a parameter sweep."""
    parser = argparse.ArgumentParser(
        description="Run a GARLAND parameter sweep from a YAML/TOML config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        required=True,
        help="Path to sweep definition (YAML or TOML)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for sweep results (overrides config file value)",
    )
    parser.add_argument(
        "--write-run-outputs",
        action="store_true",
        help="Write per-run CSV/JSON outputs under the sweep output directory",
    )
    return parser.parse_args(argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the default simulation run."""
    return parse_run_args(argv)


def _print_summary_table(results) -> None:
    display_columns = [
        column
        for column in results.columns
        if column.startswith(
            ("run_", "param_", "total_epsilon", "time_to_detection", "fpr_", "fnr_")
        )
    ]
    print(results[display_columns].to_string(index=False))


def main_sweep(argv: list[str] | None = None) -> None:
    """Run a parameter sweep experiment."""
    args = parse_sweep_args(argv)
    results = run_sweep(
        args.sweep_config,
        output_dir=args.output_dir,
        write_run_outputs=args.write_run_outputs,
    )
    print("Sweep complete")
    print("=" * 50)
    _print_summary_table(results)
    output_dir = Path(args.output_dir) if args.output_dir else Path("output/sweep")
    print(f"\nResults CSV: {output_dir / 'sweep_results.csv'}")


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the GARLAND simulation."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "sweep":
        main_sweep(argv[1:])
        return

    args = parse_run_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config_from_args(args)
    active_attacks = config.attacks.active_attacks

    print("GARLAND Epidemiological Security Testbed")
    print("=" * 50)
    print(f"Population: {config.n_agents:,} agents")
    print(f"Wearable penetration: {config.wearable_fraction*100:.1f}%")
    print(f"Simulation: {config.n_steps} steps ({config.n_steps * 5 / 60:.1f} hours)")
    print(
        f"Spatial: {config.grid_width:.0f}m × {config.grid_height:.0f}m, "
        f"{config.spatial_backend} index"
    )
    print(f"Mobility: {config.mobility_model} ({config.mobility_speed_m:.0f} m/step max)")
    print(f"Biometrics: {config.biometric_synthesis} synthesis")
    print(f"Privacy: ε={config.privacy.epsilon_per_response}/response, K={config.privacy.k_min}")
    print(f"Attacks: {[a.value for a in active_attacks] if active_attacks else 'None'}")
    if args.config:
        print(f"Config: {args.config}")
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
