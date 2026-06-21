---
name: garland-testing
description: Run, write, and debug tests for the GARLAND epidemiological security testbed. Use when running pytest, adding tests, fixing test failures, checking coverage, or validating simulation and privacy behavior.
paths:
  - "tests/**"
  - "src/garland/**"
  - "pyproject.toml"
  - ".github/workflows/**"
---

# GARLAND Testing

## When to Use

- Running or debugging the test suite
- Adding tests for simulation, privacy, attacks, or metrics
- Verifying a fix before opening a PR

## Environment

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v
python3 -m pytest tests/ --cov=garland --cov-report=term-missing
ruff check src tests
mypy
```

**Current:** 110 tests, ~91% coverage, CI on Python 3.10/3.12.

## Test Files

| File | Focus |
|------|-------|
| `test_simulation.py` | Model init, SEIR, plume, biometrics, detection classification, attacks summary, protocol integration |
| `test_privacy.py` | Planar Laplace, RR, dilution, aggregator, protocol integration |
| `test_attacks.py` | AttackOrchestrator, eclipse, replay, correlation |
| `test_cli.py` | CLI argument parsing and smoke runs |
| `test_metrics.py` | Episode-granular FN/TN counting |
| `test_scaling.py` | Vectorized init, cell ID cache, benchmark module |

## Key Fixtures

### `small_config` (`test_simulation.py`)

1000 agents, 50 steps — use for full `GarlandModel.run()`.

### `medium_config` (`test_scaling.py`)

5000 agents — scaling smoke tests.

### `rng`, `populated_grid` (`test_privacy.py`)

Seeded RNG and 1000-agent spatial grid for dilution tests.

## Writing Tests

1. Use `seed=42` or `np.random.default_rng(42)`
2. Assert meaningful bounds — not just `is not None`
3. Avoid conditional skips without guaranteed fixtures
4. Never use 250K agents in CI tests

## Regression Targets (open bugs)

When fixing issues, add tests:

| Issue | Test focus |
|-------|------------|
| #25 | FEBRILE/MULTI_SYSTEM zone-local TP (mirror `TestDetectionClassification`) |
| #24 | Summary ε matches adaptive composition when wired |

## Integration Tests (existing)

- `TestProtocolSimulationIntegration` — anomaly cluster → broadcast → responses
- `TestDetectionClassification` — zone-local plume/cardiac TP/FP
- `TestAttackSummaryMetrics` — all attack types update summary
- `TestEpisodeFalseNegatives` — one FN per undetected episode

## Pre-PR Checklist

- [ ] `python3 -m pytest tests/ -v` passes
- [ ] `ruff check src tests` passes
- [ ] Bug fixes include regression test
- [ ] Full suite completes in < 30s on CI runner

## CI

`.github/workflows/tests.yml`:

```yaml
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Coverage enforced via `pyproject.toml` `addopts = "--cov=garland ..."`. No fail threshold yet.

## References

- Open bugs: `../garland-issues/references/known-issues.md`
- Architecture: `../garland-architecture/SKILL.md`
