"""Tests for the operator-run disambiguation evaluation harness."""

from pathlib import Path

from scripts.disambiguation_ask_eval import ROOT, _resolve_output_path


def test_relative_eval_output_path_is_resolved_once() -> None:
    assert _resolve_output_path(Path("output/ask_eval.json")) == (ROOT / "output" / "ask_eval.json")
