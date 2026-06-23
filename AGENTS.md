# AGENTS.md — AI Agent Guidelines for GARLAND

## Repository Purpose
City-scale (250K agent) ABM testbed for evaluating a decentralized differential
privacy protocol for wearable health surveillance. Simulates SEIR epidemics +
chemical plume hazards with biometric anomaly detection, encrypted token
aggregation, K-anonymity broadcasting, and five adversarial attack types.

## Setup
```bash
pip install -e ".[dev]"              # core + pytest, ruff, mypy
pip install -e ".[dev,biosignals]"   # + NeuroKit2/scipy for optional synthesis
```

## Validation Commands
Run these before committing:
```bash
ruff check src tests
mypy
python -m pytest tests/ -v
```

## Architecture Rules
- **Simulation testbed only** — do not claim formal DP proofs or real encryption
- **Broadcast-and-filter** — no mesh routing; responses are zone-local only
- **Zone-local detection** — TP/FP classification is per hazard instance within zone
- **Both spatial backends** — H3 hex and rectangular must both work for any spatial change
- **Cell IDs rebuild after mobility** — `wearable_agents_by_cell` must stay in sync
- **Never modify tests to make them pass** — fix the implementation

## Key Files
| File | Purpose |
|------|---------|
| `src/garland/simulation.py` | `GarlandModel` — main step orchestration |
| `src/garland/app.py` | CLI (`garland`, `garland sweep`) |
| `src/garland/config.py` | YAML/TOML config loading |
| `src/garland/experiment.py` | Parameter sweep runner |
| `src/garland/spatial.py` | `H3HexGrid`, `RectangularGrid`, `create_spatial_grid` |
| `src/garland/privacy.py` | Token aggregation, K-anonymity, DP responses |
| `src/garland/agents.py` | Agent state and wearable attributes |
| `src/garland/hazards.py` | SEIR + Gaussian plume models |
| `src/garland/attacks.py` | Five attack types (sybil, eclipse, replay, deanon, correlation) |
| `src/garland/metrics.py` | Episode FPR/FNR, attack counters, instance TP |
| `src/garland/biometric_synthesis.py` | Custom + optional NeuroKit2 synthesis |
| `tests/` | 168 tests covering all modules |

## Step Pipeline (10 phases)
1. Mobility — random walk; rebuild spatial cell membership
2. SEIR — vectorized proximity S→E
3. Plume(s) — Gaussian concentration per agent
4. Biometrics — wearable anomaly synthesis
5. Tokens — anomaly → `EncryptedToken(cell_id, …)`
6. Attacks — eclipse filter, sybil/replay inject
7. Aggregate — threshold → `dilated_zone` → broadcast
8. Responses — RR + Laplace; indexed by `wearable_agents_by_cell`
9. Classify — zone-local TP/FP per hazard instance
10. Metrics — episode FPR/FNR, attack counters

## Running Simulations
```bash
garland --n-agents 1000 --n-steps 48
garland --config examples/quick.yaml
garland --spatial-backend h3 --h3-resolution 9
garland --spatial-backend rect --static-agents
garland --enable-sybil --enable-replay --enable-deanon --enable-correlation --enable-eclipse
garland sweep --sweep-config examples/privacy_sweep.yaml
```

## Code Conventions
- Python 3.10+ with strict typing (mypy strict mode)
- Ruff for linting and formatting
- Tests use pytest with coverage
- YAML/TOML config files (CLI overrides file values)
- Mesa thin inheritance only — no Mesa scheduler

## PR Requirements
- All ruff checks pass
- mypy passes
- All 168 tests pass
- Regression tests for bug fixes (see `resolved-issues.md`)
- Both spatial backends tested if touching spatial/dilution logic
- Update CHANGELOG.md for shipped features
