---
name: garland-architecture
description: Understand GARLAND architecture — simulation loop, spatial backends, mobility, multi-hazard, config, and sweeps. Use when navigating or planning changes.
paths:
  - "src/garland/**"
  - "README.md"
  - "docs/**"
  - "examples/**"
---

# GARLAND Architecture

## Purpose

City-scale (250K default) ABM testbed: SEIR + plume hazards, wearable biometrics, decentralized DP protocol, five attack types.

## High-Level Flow

```
garland [--config file] → SimulationConfig → GarlandModel.step() × N
garland sweep → experiment.run_sweep() → sweep_results.csv
```

## Step Pipeline

1. **Mobility** — optional random walk; rebuild spatial cell membership
2. **SEIR** — vectorized + proximity S→E (capped infectious checks)
3. **Plume(s)** — Gaussian concentration per agent (multi-plume supported)
4. **Biometrics** — wearables only; custom or NeuroKit2 synthesis
5. **Tokens** — anomaly → `EncryptedToken(cell_id, …)`
6. **Attacks** — eclipse filter, sybil/replay inject
7. **Aggregate** — threshold → `dilated_zone` → broadcast
8. **Responses** — RR + Laplace; indexed by `wearable_agents_by_cell`
9. **Classify** — zone-local TP/FP per hazard instance
10. **Metrics** — episode FPR/FNR, attack counters

## Spatial Backends (`spatial.py`)

| Backend | CLI | Class |
|---------|-----|-------|
| H3 hex (default) | `--spatial-backend h3` | `H3HexGrid` |
| Rectangular | `--spatial-backend rect` | `RectangularGrid` |

Factory: `create_spatial_grid(config)`. Privacy protocol uses opaque **cell IDs** from `SpatialIndex`.

## Mobility

Enabled by default. Disable with `--static-agents`. Cell IDs and `wearable_agents_by_cell` update when agents move.

## Multi-Hazard

- Multiple `PlumeConfig` entries in config file
- Multiple outbreak seeds in SEIR config
- Detection tracks `hazard_instance_id`

## Config System

- `config.py` — load YAML/TOML → `SimulationConfig`
- `examples/` — quick runs, privacy sweeps, multi-hazard
- CLI flags override file values

## Attacks (all wired)

Sybil, deanon, correlation, eclipse, replay — see README Layer 4.

## Key Modules

| Module | Notes |
|--------|-------|
| `simulation.py` | Orchestration |
| `experiment.py` | Sweeps |
| `biometric_synthesis.py` | Pluggable synthesis |
| `openwearables.py` | Export format |
| `metrics.py` | Episode metrics + instance TP counts |

## Mesa

Thin inheritance only — no Mesa scheduler.

## References

- `docs/SCALING.md`, `docs/BIOMETRICS.md`
- `garland-privacy-protocol` skill
