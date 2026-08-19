---
name: garland-testing
description: Run and write tests for GARLAND. Use when running pytest or mypy, adding regression tests, or validating simulation, privacy, mobility, config, and sweep behavior.
paths:
  - "tests/**"
  - "src/garland/**"
  - "pyproject.toml"
  - ".github/workflows/**"
---

# GARLAND Testing

## Commands

```bash
pip install -e ".[dev,biosignals]"

python -m pytest tests/ -v                    # canonical full-suite command
python -m pytest tests/test_mobility.py -v
python -m pytest tests/test_config.py -v
python -m pytest tests/test_experiment.py -v
python -m pytest tests/ --cov=garland --cov-report=term-missing

ruff check src tests
mypy
```

## CI (`.github/workflows/tests.yml`)

| Job | Steps |
|-----|-------|
| `lint` | `ruff check`, `mypy` |
| `test` | `pip install -e ".[dev,biosignals]"`, `pytest -v` on 3.10 & 3.12 |

## Test Files

| File | Covers |
|------|--------|
| `test_simulation.py` | Model, SEIR, plume, detection, attacks, protocol E2E |
| `test_privacy.py` | DP mechanisms, dilution, aggregator, integration |
| `test_attacks.py` | Orchestrator, eclipse, replay, correlation |
| `test_cli.py` | CLI, config loading |
| `test_metrics.py` | Episode FN/TN/FPR |
| `test_scaling.py` | Benchmark, init perf |
| `test_mobility.py` | Agent movement, cell rebuild |
| `test_config.py` | YAML/TOML config round-trip |
| `test_experiment.py` | Parameter sweeps |
| `test_multi_hazard.py` | Multiple plumes/outbreaks |
| `test_spatial.py` | H3 and rectangular backends |
| `test_biometric_synthesis.py` | Custom + NeuroKit2 paths |
| `test_operational_detection.py` | Null/staged operating points, detector modes, and daily alert metrics |

Measurement coverage should distinguish:

- instant versus sequential detector behavior, including warm-up suppression,
  hysteresis clearing, and continued emission during active episodes;
- zone-local hazard detections versus provenance-only attributed and
  coincidental detections;
- affected-agent token fragmentation and the largest same-zone/type group
  relative to `threshold_m`;
- `None` for no evidence or no denominator, rather than a flattering zero or
  one.

## Fixtures

- `small_config` — 1000 agents, 50 steps (`test_simulation.py`)
- `medium_config` — 5000 agents (`test_scaling.py`)
- `rng`, `populated_grid` — privacy/spatial tests

**Do not run 250K-agent tests in CI.** Use `python -m garland.benchmark` locally.

## Writing Tests

- Seed RNGs (`seed=42`)
- Assert bounded values, not bare `is not None`
- Avoid conditional skips without guaranteed fixtures
- **Bug fixes require regression tests**

## Pre-PR Checklist

- [ ] `ruff check src tests`
- [ ] `mypy`
- [ ] `python -m pytest tests/ -v`

## Runtime (end-to-end) validation of metrics blocks

Unit tests do not prove a new summary block behaves at scale. For metrics work
(e.g. `detection_power`), run the shipped scenarios and validate the JSON:

```bash
PYTHONPATH=src uv run --no-sync --no-build garland --config examples/<scenario>.yaml \
  --no-plots --output-dir output/<name>
PYTHONPATH=src uv run --no-sync --no-build garland sweep --sweep-config examples/<sweep>.yaml
```

Then run a short Python invariant script over every `output/**/summary.json`:
finiteness, rates in `[0, 1]` or `None`, non-negative latencies, and any
conservation identity the block claims (totals equal the sum of their strata).

Observed wall times on a 2-core box (1728 steps, `detection_power_town.yaml`):
2K ≈ 1m20s, 10K ≈ 7m, 25K ≈ 18m; peak RSS stayed under ~500 MB even at 25K, so
population ladders up to 25K are practical in a session, 250K is not.

Gotchas worth knowing before you plan a run:

- `garland sweep` writes `sweep_results.csv` into the sweep config's
  `output_dir`, and `run_sweep` reports that directory back in
  `results.attrs["output_dir"]`.
- Sweeps write per-run `summary.json` files only under `--write-run-outputs`;
  without it, invariant checks for a sweep have to run against the CSV columns.
- A CLI flag whose "off" value equals its argparse default cannot override a
  value set in YAML. Such flags should default to `None`; if you meet one that
  does not, fix the flag rather than working around it with a copied config.
- Stdout is block-buffered when piped, so a long run shows no progress through
  `tee`. Poll process RSS/etime with `ps` instead, or run under
  `/usr/bin/time -v` for peak RSS.
- `sweep_results.csv` carries no `warranted_detections`, `detection_event_counts`
  or `unexplained_detection_rate` column, so a claim about any of those needs
  per-arm single runs (or `--write-run-outputs`) rather than a sweep.
- `detection_event_counts.disease_true_positive` and
  `attributed_disease_detections` measure different things and can disagree on the
  same run. Name which key a reported figure came from.
- Run cost tracks wearers, not residents: 2K at `wearable_fraction` 0.15 ≈ 1.5 min
  against ≈ 6-8 min at 0.6, the same as 8K at 0.15.
- To check that a sweep test tests its mechanism, edit the swept list in the
  temporary YAML (flatten it, then invert it) and confirm the test goes red both
  times, then restore with `git checkout --`.

## References

- `../garland-issues/references/resolved-issues.md`
- `CONTRIBUTING.md`
