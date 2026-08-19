"""Measure benign residual variance for the cold-start covariance prior.

The measurement uses the model's own benign physiology: devices mature for five
days and residuals are collected on day six with the live activity level and
jitter. The result is a calibration harness, not a claim about real wearables.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from garland.biometrics import BaselineTracker, generate_observation, generate_profiles
from garland.channels import DEFAULT_CHANNEL_SET as CHANNELS
from garland.constants import STEPS_PER_DAY
from garland.paths import resolve_under_base, write_json_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("output/coldstart_variance_calibration.json")
SEED = 20260819
N_DEVICES = 100
MATURATION_DAYS = 5


def _activity(hour_of_day: float) -> float:
    """Return the live-path activity contribution for one hour."""
    if 6 <= hour_of_day <= 22:
        return float(0.3 * max(0.0, math.sin(math.pi * (hour_of_day - 6) / 12)))
    return 0.0


def _resolve_output_path(user_path: Path) -> Path:
    """Resolve an output argument beneath the repository output directory."""
    output_base = ROOT / "output"
    relative_path = user_path
    if not user_path.is_absolute() and user_path.parts[:1] == ("output",):
        relative_path = Path(*user_path.parts[1:])
    return resolve_under_base(output_base, relative_path)


def measure(
    *,
    seed: int = SEED,
    n_devices: int = N_DEVICES,
    maturation_days: int = MATURATION_DAYS,
) -> dict[str, object]:
    """Measure mature benign residual variances and cold-start driver shares."""
    if n_devices < 1:
        raise ValueError("n_devices must be positive")
    if maturation_days < 1:
        raise ValueError("maturation_days must be positive")

    rng = np.random.default_rng(seed)
    profiles = generate_profiles(n_devices, rng, CHANNELS)
    names = [channel.name for channel in CHANNELS.channels]

    residuals: list[np.ndarray] = []
    for device_index in range(n_devices):
        device_rng = np.random.default_rng([seed, device_index])
        tracker = BaselineTracker()
        # Mature the tracker, then collect residuals on the following day.
        for step in range(STEPS_PER_DAY * (maturation_days + 1)):
            hour_of_day = (step * 5 / 60.0) % 24
            observation = CHANNELS.clamp(
                generate_observation(
                    profiles[device_index],
                    hour_of_day,
                    day_of_year=196,
                    rng=device_rng,
                    activity_level=_activity(hour_of_day) + device_rng.normal(0, 0.05),
                )
            )
            hour = int(hour_of_day) % 24
            if step >= STEPS_PER_DAY * maturation_days:
                residuals.append(observation - tracker.expected_baseline(hour, 7))
            tracker.update(observation, hour, 7)

    matrix = np.asarray(residuals)
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(names):
        prior = float(CHANNELS.channels[index].prior_variance)
        variance = float(matrix[:, index].var())
        rows.append(
            {
                "channel": name,
                "prior_variance": prior,
                "benign_residual_variance": variance,
                "benign_residual_sd": math.sqrt(variance),
                "prior_to_measured_ratio": prior / variance,
            }
        )

    def driver_shares(prior_variances: np.ndarray) -> dict[str, float]:
        scaled = np.abs(matrix) / np.sqrt(prior_variances)
        winners = scaled.argmax(axis=1)
        return {name: float((winners == index).mean()) for index, name in enumerate(names)}

    driver_shares_by_prior = driver_shares(CHANNELS.prior_variances)
    flat_prior = np.full(len(CHANNELS), 10.0, dtype=np.float64)
    return {
        "seed": seed,
        "n_devices": n_devices,
        "maturation_days": maturation_days,
        "rows": rows,
        "cold_start_driver_shares": driver_shares_by_prior,
        "flat_prior_driver_shares": driver_shares(flat_prior),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path (default: output/coldstart_variance_calibration.json)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--devices", type=int, default=N_DEVICES)
    parser.add_argument("--maturation-days", type=int, default=MATURATION_DAYS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the calibration and print each measured variance against its prior."""
    args = _parser().parse_args(argv)
    result = measure(
        seed=args.seed,
        n_devices=args.devices,
        maturation_days=args.maturation_days,
    )
    write_json_file(_resolve_output_path(args.output), result)
    print(f"{'channel':20s} {'prior var':>10s} {'benign var':>12s} {'ratio':>8s}")
    for row in result["rows"]:
        print(
            f"{row['channel']:20s} {row['prior_variance']:10.2f} "
            f"{row['benign_residual_variance']:12.3f} "
            f"{row['prior_to_measured_ratio']:8.2f}"
        )
    print("\nper-channel share of cold-start driver maxima:")
    for name, share in result["cold_start_driver_shares"].items():
        print(f"  {name:20s} {share:6.1%}")


if __name__ == "__main__":
    main()
