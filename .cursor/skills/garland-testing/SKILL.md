---
name: garland-testing
description: Run, write, and debug tests for the GARLAND epidemiological security testbed. Use when running pytest, adding tests, fixing test failures, checking coverage, or validating simulation and privacy behavior.
paths:
  - "tests/**"
  - "src/garland/**"
  - "pyproject.toml"
---

# GARLAND Testing

## When to Use

- Running or debugging the test suite
- Adding tests for simulation, privacy, hazards, or attacks
- Verifying a fix before opening a PR
- Writing integration tests for the privacy protocol pipeline

## Environment Setup

```bash
pip install -e ".[dev]"
```

Run from repository root (`/workspace` or project root).

## Running Tests

```bash
# Full suite
python3 -m pytest tests/ -v

# Single file or test
python3 -m pytest tests/test_privacy.py -v
python3 -m pytest tests/test_simulation.py::TestEndToEnd::test_simulation_runs_without_error -v

# With coverage
python3 -m pytest tests/ --cov=garland --cov-report=term-missing

# Lint (run before committing)
ruff check src tests
```

## Test Layout

| File | Focus |
|------|-------|
| `tests/test_simulation.py` | Model init, SEIR, plume, biometrics, end-to-end smoke |
| `tests/test_privacy.py` | Planar Laplace, RR, spatial dilution, attacks (unit level), aggregator |

There is no `conftest.py`; fixtures are defined per file.

## Shared Fixtures

### `small_config` (`test_simulation.py`)

Reduced-scale `SimulationConfig` for fast runs:

- 1000 agents, 50 steps, plume at step 10
- Use for any test that runs `GarlandModel.run()`

### `rng` / `populated_grid` (`test_privacy.py`)

- Seeded NumPy generators
- 1000 agents on a 2000×2000 m grid for spatial dilution tests

When adding simulation tests, reuse `small_config` or create a similarly small config. **Never default to 250K agents in tests.**

## Writing Tests

### Conventions

1. Match existing style: pytest classes (`TestSEIR`), descriptive docstrings, `from __future__ import annotations`
2. Prefer deterministic tests: pass `seed=42` or use `np.random.default_rng(42)`
3. Assert meaningful bounds — avoid `assert error is not None` without a threshold
4. Avoid conditional skips like `if sparse_cell is not None:` without guaranteeing the condition; use fixtures that force sparse/dense cells

### Unit vs integration

| Layer | What to test | Example |
|-------|--------------|---------|
| Privacy primitives | `privacy.py` functions in isolation | `planar_laplace_noise`, `randomized_response` |
| Spatial | `SpatialGrid.dilated_zone` with real cell IDs | `test_privacy.py::TestSpatialDilution` |
| Hazards | SEIR transitions, plume concentration patterns | `test_simulation.py::TestSEIR`, `TestPlume` |
| Integration | Full `GarlandModel.step()` pipeline | **Gap — add when fixing zone ID mismatch (#5)** |

### Critical integration gap (issue #12)

The suite does **not** yet verify:

```
anomaly token → aggregator threshold → spatial dilution → broadcast → agent response → detection
```

When fixing protocol bugs, add an integration test that:

1. Places several wearable agents in the same grid cell with forced anomalies
2. Steps until `broadcasts_issued > 0`
3. Asserts agents in the dilated zone respond
4. Uses **grid cell IDs**, not neighborhood IDs, once #5 is fixed

## Known Test Pitfalls

1. **Mesa import requires `networkx`** — test collection fails without it
2. **Spatial dilution tests use cell IDs** — token/zone tests must use the same namespace as production code (see `garland-privacy-protocol` skill, issue #5)
3. **End-to-end tests pass even when protocol matching is broken** — smoke tests only check no crash and column presence
4. **Attack tests are unit-level only** — `AttackOrchestrator` is not exercised in simulation tests

## Pre-PR Checklist

- [ ] `python3 -m pytest tests/ -v` passes
- [ ] `ruff check src tests` passes (if ruff installed)
- [ ] New behavior has tests; bug fixes include a regression test
- [ ] Tests run in reasonable time (< 30s for full suite on CI-sized runner)

## References

- Test configuration: `pyproject.toml` → `[tool.pytest.ini_options]`
- Ruff config: `pyproject.toml` → `[tool.ruff]`
- Known coverage gaps: `.cursor/skills/garland-issues/references/known-issues.md`
