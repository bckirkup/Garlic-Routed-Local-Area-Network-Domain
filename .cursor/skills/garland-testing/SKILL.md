---
name: garland-testing
description: Run, write, and debug tests for GARLAND. Use when running pytest, adding tests, fixing failures, checking coverage, or validating simulation, privacy, attack, and metrics behavior.
paths:
  - "tests/**"
  - "src/garland/**"
  - "pyproject.toml"
  - ".github/workflows/**"
---

# GARLAND Testing

## When to Use

- Running or debugging the test suite
- Adding regression tests for bug fixes
- Validating privacy protocol or attack changes

## Setup & Commands

```bash
pip install -e ".[dev]"

# Full suite (110 tests, ~18s, ~91% cov)
python3 -m pytest tests/ -v

# Single module
python3 -m pytest tests/test_simulation.py -v
python3 -m pytest tests/test_privacy.py::TestProtocolIntegration -v

# Coverage detail
python3 -m pytest tests/ --cov=garland --cov-report=term-missing

# Lint (run before PR)
ruff check src tests
```

## Test Layout

| File | Covers |
|------|--------|
| `test_simulation.py` | Model init, SEIR, plume, biometrics, detection classification, attack summary, protocol E2E |
| `test_privacy.py` | Planar Laplace, RR, dilution, aggregator, deanonymization, protocol integration |
| `test_attacks.py` | Sybil, eclipse, replay, correlation, orchestrator |
| `test_cli.py` | CLI parsing and smoke execution |
| `test_metrics.py` | Episode-granular FN/TN/FPR logic |
| `test_scaling.py` | Vectorized init, cell IDs, benchmark helper, SEIR cap config |

## Fixtures

| Fixture | File | Use |
|---------|------|-----|
| `small_config` | `test_simulation.py` | 1000 agents, 50 steps — default for `GarlandModel.run()` |
| `medium_config` | `test_scaling.py` | 5000 agents — perf smoke |
| `rng` | `test_privacy.py` | Seeded `np.random.default_rng(12345)` |
| `populated_grid` | `test_privacy.py` | 1000 agents on 2000×2000 m grid |

**Never use 250K agents in CI tests.** Use `python -m garland.benchmark` for scale validation locally.

## Writing Tests

### Conventions

- `from __future__ import annotations`
- pytest classes: `class TestFeatureName:`
- Docstring per test explaining what is validated
- Seed RNGs: `seed=42` or `np.random.default_rng(42)`

### Assertion quality

```python
# Good — bounded claim
assert error is None or error > 50.0

# Bad — always passes when error exists
assert error is not None

# Bad — silent skip
if sparse_cell is not None:
    assert len(zone) > 1
```

Force fixture preconditions instead of conditional skips.

### Required for bug fixes

Every **bug fix** PR must add a regression test that would fail on the broken behavior.

## Key Test Areas

### Privacy protocol integration

`TestProtocolSimulationIntegration` — forced anomalies → broadcast → responses.

### Zone-local detection

`TestDetectionClassification` — plume/cardiac/febrile TP/FP use zone ground truth (not global timestep).

### Episode metrics

`TestEpisodeFalseNegatives` — at most one FN per undetected hazard episode.

### Attack metrics

`TestAttackSummaryMetrics` — each `--enable-*` flag updates corresponding summary fields.

## CI

`.github/workflows/tests.yml`:

- Matrix: Python 3.10, 3.12
- `pip install -e ".[dev]"` then `python -m pytest tests/ -v`
- Coverage via `pyproject.toml` `addopts = "--cov=garland ..."`

## Pre-PR Checklist

- [ ] `python3 -m pytest tests/ -v` passes
- [ ] `ruff check src tests` passes
- [ ] Bug fix includes regression test
- [ ] No tests added at 250K scale

## References

- Regression checklist: `../garland-issues/references/resolved-issues.md`
- Architecture: `../garland-architecture/SKILL.md`
