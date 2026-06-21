---
name: garland-development
description: Set up, build, lint, and contribute to the GARLAND Python package. Use when installing dependencies, running the CLI, making code changes, committing, or opening pull requests.
paths:
  - "src/**"
  - "pyproject.toml"
  - "README.md"
---

# GARLAND Development

## When to Use

- First-time environment setup
- Running the simulation CLI
- Making code changes and opening PRs
- Linting and packaging

## Project Overview

**GARLAND** (Garlic-Routed Local Area Network Domain) is a privacy-preserving epidemiological security testbed: agent-based simulation of SEIR disease + environmental plume hazards with a decentralized differential-privacy protocol.

- **Language:** Python ≥ 3.10
- **Package:** `src/garland/` (Hatchling build)
- **CLI:** `garland` → `garland.app:main`
- **Version:** `0.1.0`

## Setup

```bash
git clone https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain.git
cd Garlic-Routed-Local-Area-Network-Domain
pip install -e ".[dev]"
```

Verify:

```bash
python3 -m pytest tests/ -v
garland --n-agents 1000 --n-steps 48 --no-plots
```

## Running the Simulation

```bash
# Quick test (1000 agents, 4 hours simulated)
garland --n-agents 1000 --n-steps 48

# Full scale (slow — 250K agents, 7 days)
garland --n-agents 250000 --n-steps 2016

# With Sybil attack
garland --n-agents 50000 --enable-sybil --sybil-count 30

# Custom privacy params
garland --epsilon-per-response 0.05 --k-min 100 --laplace-scale 300
```

Outputs go to `output/` (gitignored):

- `simulation_metrics.csv`
- `summary.json`
- `*.png` plots (unless `--no-plots`)

## Code Conventions

Follow existing patterns in `src/garland/`:

1. **Dataclasses** for config and state (`SimulationConfig`, `PrivacyConfig`, etc.)
2. **NumPy vectorization** for population-scale state; Python loops only for wearable agents (~15% of population)
3. **Module docstrings** at top of each file describing layer purpose
4. **Minimal scope** — focused diffs; no drive-by refactors
5. **Line length:** 100 (Ruff)
6. **Imports:** `from __future__ import annotations`; sorted imports (Ruff `I`)

### Do not add without request

- Pydantic models (project uses dataclasses)
- Heavy abstractions or one-line helper wrappers
- Tests that only assert the obvious

## Linting

```bash
ruff check src tests
ruff format src tests  # if formatting needed
```

Ruff rules: `E`, `F`, `W`, `I` on `src` and `tests`.

## Git Workflow

```bash
# Branch from main
git checkout main && git pull origin main
git checkout -b cursor/my-change-b383

# After changes
git add -p
git commit -m "Fix zone ID mismatch in privacy protocol"
git push -u origin cursor/my-change-b383
```

**Branch naming (Cloud Agent):** `cursor/<descriptive-name>-b383`

**Commit messages:** Complete sentences; state what changed and why.

**PRs:** Draft by default; link issues with `Closes #N`. See `garland-issues` skill.

## Module Map

| Module | Responsibility |
|--------|----------------|
| `app.py` | CLI, config assembly, output |
| `simulation.py` | `GarlandModel` orchestration, main step loop |
| `agents.py` | `CitizenAgent`, `NetworkAggregator` |
| `biometrics.py` | Synthetic vitals, baselines, Mahalanobis anomaly |
| `hazards.py` | SEIR engine, Gaussian plume |
| `privacy.py` | DP mechanisms, tokens, aggregator state |
| `spatial.py` | Cell grid, K-anonymity dilution |
| `attacks.py` | Adversarial models (partially wired) |
| `metrics.py` | Evaluation metrics, CSV/plots |

Deep architecture: use `garland-architecture` skill.

## Dependencies Note

Several unused dependencies were removed in v0.1.1. Mesa is used minimally (subclass only). Do not add new unused dependencies.

## Related Skills

| Skill | Use when |
|-------|----------|
| `garland-testing` | Writing or running tests |
| `garland-issues` | Fixing backlog issues |
| `garland-architecture` | Understanding data flow |
| `garland-privacy-protocol` | Changing privacy/spatial logic |
