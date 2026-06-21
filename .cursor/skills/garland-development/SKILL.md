---
name: garland-development
description: Set up, build, lint, and contribute to the GARLAND Python package. Use when installing dependencies, running the CLI, making code changes, committing, or opening pull requests.
paths:
  - "src/**"
  - "pyproject.toml"
  - "README.md"
  - "docs/**"
---

# GARLAND Development

## When to Use

- Environment setup
- Running simulation or benchmarks
- Making code changes and opening PRs

## Project Overview

**GARLAND** — privacy-preserving epidemiological security testbed (Python ≥ 3.10, v0.1.0).

- **Package:** `src/garland/`
- **CLI:** `garland` → `garland.app:main`
- **Benchmark:** `python -m garland.benchmark`
- **License:** Apache 2.0

## Setup

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v          # 110 tests, ~91% cov
ruff check src tests                 # not in CI yet (#31)
```

## Running

```bash
# Quick run
garland --n-agents 1000 --n-steps 48

# Full city scale
garland --n-agents 250000 --n-steps 2016 --no-plots

# Attacks (all five types supported)
garland --n-agents 5000 --enable-sybil --enable-replay --enable-deanon

# Benchmark before large runs
python -m garland.benchmark --quick
python -m garland.benchmark --n-agents 250000 --n-steps 30
```

Outputs: `output/` (CSV, JSON summary, PNG plots).

## Code Conventions

- Dataclasses for config; NumPy for population-scale state
- Python loops only over wearable agents (~15% of N)
- Module docstrings; line length 100; Ruff E/F/W/I
- Minimal scope — no drive-by refactors

## Git Workflow

```bash
git checkout main && git pull origin main
git checkout -b cursor/my-change-b383
# ... changes ...
git push -u origin cursor/my-change-b383
```

PR body: `Closes #N` + issue type (Bug fix / Feature / etc.)

## Module Map

| Module | Role |
|--------|------|
| `app.py` | CLI |
| `simulation.py` | `GarlandModel` step loop |
| `agents.py` | `CitizenAgent`, `NetworkAggregator` |
| `attacks.py` | `AttackOrchestrator` + 5 attack types |
| `biometrics.py` | Synthetic vitals, baselines |
| `hazards.py` | SEIR + plume |
| `privacy.py` | DP mechanisms, tokens |
| `spatial.py` | Cell grid, K-anonymity dilution |
| `metrics.py` | Episode metrics, CSV, plots |
| `benchmark.py` | Scaling benchmark helper |

## Dependencies

Runtime: `mesa`, `networkx`, `numpy`, `pandas`, `matplotlib`

Removed unused deps (neurokit2, scipy, h3, pydantic) — see closed #10.

## CI

`.github/workflows/tests.yml` — pytest on 3.10 and 3.12 with coverage.

Not yet: ruff (#31), mypy (#37).

## Open work

See `../garland-issues/references/known-issues.md` (bugs) and `feature-backlog.md` (features).

## Related Skills

| Skill | Use when |
|-------|----------|
| `garland-testing` | Writing/running tests |
| `garland-issues` | Backlog triage |
| `garland-architecture` | Data flow |
| `garland-privacy-protocol` | Privacy changes |
