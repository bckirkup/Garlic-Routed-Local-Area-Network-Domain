"""Operator-run measurement of disambiguation ask quality.

This script runs five long-form variants of the authored evaluation scenario.
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
from garland.paths import resolve_under_base, write_json_file
from garland.simulation import SimulationConfig

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "disambiguation_evaluation.yaml"
DEFAULT_OUTPUT = ROOT / "output" / "disambiguation_ask_eval_results.json"

VARIANT_MIX_ONBOARDING = "mix+onboarding"
VARIANT_MIX_ONBOARDING_TIGHT = "mix+onboarding, tight budget"

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
    "disambiguation_max_ask_epsilon_delta",
    "disambiguation_asks_suppressed_by_budget",
    "disambiguation_precision",
    "disambiguation_precision_by_hypothesis",
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
    """Yield the five evaluation variants from the authored scenario."""
    base = load_config_file(SCENARIO)

    yield VARIANT_MIX_ONBOARDING, base

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

    tight_budget = copy.deepcopy(base)
    tight_budget.disambiguation.ask_epsilon_budget = 5.0
    yield VARIANT_MIX_ONBOARDING_TIGHT, tight_budget


def _resolve_output_path(user_path: Path) -> Path:
    """Resolve an output argument beneath the repository output directory."""
    output_base = ROOT / "output"
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
        help="JSON output path (default: output/disambiguation_ask_eval_results.json)",
    )
    parser.add_argument(
        "--variant",
        choices=(
            VARIANT_MIX_ONBOARDING,
            "mix only",
            "mix+outbreak",
            "no ground truth",
            VARIANT_MIX_ONBOARDING_TIGHT,
        ),
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
    unlimited_mix_onboarding: dict[str, Any] | None = None

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
        if name == VARIANT_MIX_ONBOARDING:
            unlimited_mix_onboarding = summary
        elif name == VARIANT_MIX_ONBOARDING_TIGHT:
            if unlimited_mix_onboarding is None:
                raise RuntimeError(
                    "The tight-budget variant requires the unlimited mix+onboarding variant"
                )
            if asks >= unlimited_mix_onboarding["disambiguation_queries_issued"]:
                raise RuntimeError("The tight budget did not reduce issued asks")
            if summary["disambiguation_asks_suppressed_by_budget"] <= 0:
                raise RuntimeError("The tight budget did not suppress any asks")
            channel_epsilon = (
                summary["disambiguation_answer_epsilon"] + summary["disambiguation_ack_epsilon"]
            )
            budget = config.disambiguation.ask_epsilon_budget
            if channel_epsilon > budget + summary["disambiguation_max_ask_epsilon_delta"]:
                raise RuntimeError("The tight budget exceeded its one-ask overshoot allowance")
        print(f"\n=== {name} ===")
        print(json.dumps(rows[name], indent=1, sort_keys=True))
        print(
            "asks_suppressed_by_budget="
            f"{summary['disambiguation_asks_suppressed_by_budget']} "
            f"precision={summary['disambiguation_precision']}"
        )
        print(
            "precision_by_hypothesis="
            + json.dumps(summary["disambiguation_precision_by_hypothesis"], sort_keys=True)
        )
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

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(output_path, rows, default=str)


if __name__ == "__main__":
    main()
