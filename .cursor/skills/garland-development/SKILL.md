---
name: garland-development
description: Set up, run, lint, and contribute to the GARLAND Python package. Use when installing deps, running the CLI or benchmark, making code changes, or opening pull requests.
paths:
  - "src/**"
  - "pyproject.toml"
  - "README.md"
  - "docs/**"
---

# GARLAND Development

## Project

**GARLAND** (Garlic-Routed Local Area Network Domain) — agent-based epidemiological security testbed with a decentralized differential-privacy protocol.

| Item | Value |
|------|-------|
| Version | 0.1.0 |
| Python | ≥ 3.10 |
| License | Apache 2.0 |
| Entry | `garland` → `garland.app:main` |
| Package | `src/garland/` |

## Setup

```bash
git clone https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain.git
cd Garlic-Routed-Local-Area-Network-Domain
pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

## Run Simulation

```bash
# Quick (1000 agents, 4 h simulated)
garland --n-agents 1000 --n-steps 48

# City scale (250K, 7 days — slow)
garland --n-agents 250000 --n-steps 2016 --no-plots

# Attacks (all five types have CLI flags)
garland --n-agents 5000 --enable-sybil --enable-replay --enable-deanon \
  --enable-correlation --enable-eclipse

# Benchmark before large runs
python -m garland.benchmark --quick
python -m garland.benchmark --n-agents 250000 --n-steps 30
```

## Output

Written to `output/` (gitignored):

- `simulation_metrics.csv` — per-step metrics
- `summary.json` — detection, privacy, attack summary
- `*.png` — SEIR, detection timeline, ε budget, protocol activity

## Code Conventions

- **Dataclasses** for all config (`SimulationConfig`, `PrivacyConfig`, …)
- **NumPy arrays** for population state; Python loops only over wearables (~15% of N)
- **Grid cell IDs** for all privacy protocol zone matching
- **Line length** 100; **Ruff** rules E/F/W/I
- **Minimal diffs** — no drive-by refactors

## Module Map

| Module | Role |
|--------|------|
| `app.py` | CLI and config assembly |
| `simulation.py` | `GarlandModel` step loop |
| `agents.py` | `CitizenAgent`, `NetworkAggregator` |
| `attacks.py` | `AttackOrchestrator` + 5 attack types |
| `biometrics.py` | Synthetic vitals, baselines, Mahalanobis |
| `hazards.py` | SEIR engine, Gaussian plume |
| `privacy.py` | DP mechanisms, tokens, aggregator state |
| `spatial.py` | Cell grid, K-anonymity dilution |
| `metrics.py` | Episode metrics, CSV, plots |
| `benchmark.py` | Optional scaling benchmark |

## Lint & Test

```bash
ruff check src tests
python3 -m pytest tests/ -v
```

## Git & PRs

```bash
git checkout -b cursor/my-feature-b383
git push -u origin cursor/my-feature-b383
```

PR must include `Closes #N` for tracked work. See `garland-issues` skill.

## Dependencies

Runtime: `mesa`, `networkx`, `numpy`, `pandas`, `matplotlib`

Dev: `pytest`, `pytest-cov`, `ruff`

## Related Skills

| Skill | When |
|-------|------|
| `garland-testing` | Tests |
| `garland-architecture` | System design |
| `garland-privacy-protocol` | Privacy changes |
| `garland-issues` | Backlog |
