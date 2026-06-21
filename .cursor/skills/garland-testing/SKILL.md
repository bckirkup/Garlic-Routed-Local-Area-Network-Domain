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

python -m pytest tests/ -v                    # 168 tests
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

## References

- `../garland-issues/references/resolved-issues.md`
- `CONTRIBUTING.md`
