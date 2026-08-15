"""Operator-run measurement of disambiguation ask quality.

This script runs four long-form variants of the authored evaluation scenario.
It is intentionally not wired into pytest or CI because the complete
measurement takes approximately 30 minutes.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from garland.adoption import AdoptionConfig
from garland.config import load_config_file
from garland.experiment import run_simulation
from garland.hazards import OutbreakSeed
from garland.simulation import SimulationConfig

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "disambiguation_evaluation.yaml"
DEFAULT_OUTPUT = ROOT / "output" / "disambiguation_ask_eval_results.json"

KEYS = (
    "total_broadcasts",
    "disambiguation_queries_issued",
    "disambiguation_well_founded_queries",
    "disambiguation_unfounded_queries",
    "disambiguation_unscored_queries",
    "disambiguation_well_founded_by_hypothesis",
    "disambiguation_unfounded_by_hypothesis",
    "disambiguation_unscored_by_hypothesis",
    "disambiguation_unfounded_ask_epsilon",
    "disambiguation_unscored_ask_epsilon",
    "disambiguation_answer_epsilon",
    "disambiguation_ack_epsilon",
    "disambiguation_yes_answers",
    "disambiguation_no_answers",
    "disambiguation_unanswered_expired",
    "disambiguation_unresolved_hypotheses",
    "total_epsilon",
    "benign_attributed_detections",
    "benign_misattributed_detections",
)


def variants() -> Iterator[tuple[str, SimulationConfig]]:
    """Yield the four evaluation variants from the authored scenario."""
    base = load_config_file(SCENARIO)

    yield "mix+onboarding", base

    fully_adopted = copy.deepcopy(base)
    fully_adopted.adoption = AdoptionConfig()
    yield "mix only", fully_adopted

    outbreak = copy.deepcopy(base)
    outbreak.seir = replace(
        outbreak.seir,
        initial_infected=15,
        outbreaks=[
            OutbreakSeed(
                outbreak_id="office_cluster",
                start_step=432,
                initial_infected=12,
                center_x=1800.0,
                center_y=1200.0,
                seed_radius=200.0,
            )
        ],
    )
    yield "mix+outbreak", outbreak

    no_truth = copy.deepcopy(base)
    no_truth.confounders = replace(no_truth.confounders, enabled=False)
    yield "no ground truth", no_truth


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path (default: output/disambiguation_ask_eval_results.json)",
    )
    parser.add_argument(
        "--variant",
        choices=("mix+onboarding", "mix only", "mix+outbreak", "no ground truth"),
        action="append",
        help="Run only this variant; may be supplied more than once.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Override the scenario step count for a smoke run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run selected variants and write their summaries as JSON."""
    args = _parser().parse_args(argv)
    selected = set(args.variant) if args.variant else None
    rows: dict[str, dict[str, Any]] = {}

    for name, config in variants():
        if selected is not None and name not in selected:
            continue
        if args.steps is not None:
            if args.steps < 1:
                raise ValueError("--steps must be positive")
            config = replace(config, n_steps=args.steps)
        summary = run_simulation(config)
        rows[name] = {key: summary.get(key) for key in KEYS}
        asks = summary["disambiguation_queries_issued"]
        buckets = (
            summary["disambiguation_well_founded_queries"]
            + summary["disambiguation_unfounded_queries"]
            + summary["disambiguation_unscored_queries"]
        )
        if asks != buckets:
            raise RuntimeError(
                f"Scoring buckets do not conserve asks for {name}: {asks} != {buckets}"
            )
        print(f"\n=== {name} ===")
        print(json.dumps(rows[name], indent=1, sort_keys=True))
        if asks:
            print(
                "well_founded_frac="
                f"{summary['disambiguation_well_founded_queries'] / asks:.3f} "
                "unfounded_frac="
                f"{summary['disambiguation_unfounded_queries'] / asks:.3f} "
                "unscored_frac="
                f"{summary['disambiguation_unscored_queries'] / asks:.3f}"
            )
            total_disambiguation_epsilon = (
                summary["disambiguation_answer_epsilon"] + summary["disambiguation_ack_epsilon"]
            )
            print(
                f"asks_per_broadcast={asks / max(summary['total_broadcasts'], 1):.3f} "
                f"disambiguation_epsilon={total_disambiguation_epsilon:.3f} "
                "share_of_total_epsilon="
                f"{total_disambiguation_epsilon / max(summary['total_epsilon'], 1e-9):.3f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
