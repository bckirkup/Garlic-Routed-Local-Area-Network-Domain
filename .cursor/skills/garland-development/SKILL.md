---
name: garland-development
description: Set up, run, lint, and contribute to GARLAND. Use when installing deps, running CLI or sweeps, loading YAML/TOML configs, or opening pull requests.
paths:
  - "src/**"
  - "pyproject.toml"
  - "README.md"
  - "CONTRIBUTING.md"
  - "examples/**"
  - "docs/**"
---

# GARLAND Development

## Setup

```bash
pip install -e ".[dev]"              # core + pytest, ruff, mypy
pip install -e ".[dev,biosignals]"   # + NeuroKit2/scipy for optional synthesis
```

Verify (matches CI):

```bash
ruff check src tests
mypy
python -m pytest tests/ -v
```

See also **`CONTRIBUTING.md`**.

## Run Simulation

```bash
# CLI flags
garland --n-agents 1000 --n-steps 48

# YAML/TOML config (CLI overrides file)
garland --config examples/quick.yaml
garland --config examples/quick.toml --n-steps 100

# City scale
garland --n-agents 250000 --n-steps 2016 --no-plots

# Spatial + mobility
garland --spatial-backend h3 --h3-resolution 9          # default
garland --spatial-backend rect --static-agents          # legacy grid, no movement

# Biometrics
garland --biometric-synthesis custom                    # default NumPy
garland --biometric-synthesis neurokit                  # requires .[biosignals]

# Attacks (all five)
garland --enable-sybil --enable-replay --enable-deanon \
  --enable-correlation --enable-eclipse

# Parameter sweep
garland sweep --sweep-config examples/privacy_sweep.yaml
```

## Output

`output/` (gitignored): `simulation_metrics.csv`, `summary.json`, plots, sweep `sweep_results.csv`.

## Module Map

| Module | Role |
|--------|------|
| `app.py` | CLI (`garland`, `garland sweep`) |
| `config.py` | YAML/TOML load/save |
| `experiment.py` | Parameter sweeps |
| `simulation.py` | `GarlandModel` |
| `spatial.py` | `H3HexGrid`, `RectangularGrid`, `create_spatial_grid` |
| `biometric_synthesis.py` | Custom + optional NeuroKit2 |
| `biometric_profiles.py` | Profile generation |
| `openwearables.py` | OpenWearables export helpers |
| `attacks.py`, `agents.py`, `privacy.py`, `hazards.py`, `metrics.py`, `benchmark.py` | Core layers |

## Conventions

- Dataclasses for config; NumPy at population scale
- Grid **cell IDs** for privacy protocol (H3 or rect)
- Line length 100; ruff E/F/W/I; mypy on `src/garland`
- Minimal diffs; regression tests for bug fixes

## Git

```bash
git checkout -b cursor/my-change-b383
git push -u origin cursor/my-change-b383
```

PR body: `Closes #N`. See `garland-issues` skill.

## Related Skills

- `garland-testing` — test layout
- `garland-architecture` — system design
- `garland-issues` — backlog
