---
name: garland-architecture
description: Understand GARLAND simulation architecture, data flow, and module boundaries. Use when navigating the codebase, planning changes, debugging the step loop, or explaining how hazards, biometrics, and privacy interact.
paths:
  - "src/garland/**"
  - "README.md"
  - "docs/**"
---

# GARLAND Architecture

## When to Use

- Orienting in the codebase before a change
- Planning where to implement a feature
- Debugging unexpected simulation behavior
- Reviewing PRs that touch multiple modules

## System Purpose

Simulate a town of agents (default 250K) at 5-minute resolution with:

1. **Co-occurring hazards** — SEIR respiratory disease + Gaussian plume toxin
2. **Wearable biometrics** — ~15% of agents with Mahalanobis anomaly detection
3. **Privacy protocol** — blind gating → threshold aggregation → K-anonymity dilution → broadcast → DP responses
4. **Attack evaluation** — Sybil, deanon, correlation, eclipse, replay (all wired)

## Layer Model

```
CLI (app.py) → GarlandModel (simulation.py)
  ├── hazards (SEIR + plume)
  ├── biometrics (profiles, baselines)
  ├── spatial (cell grid, dilated zones)
  ├── privacy (tokens, DP mechanisms)
  ├── attacks (AttackOrchestrator)
  └── metrics (episode-granular FPR/FNR, CSV, plots)
```

## Simulation Step Pipeline

| Step | Action | Module |
|------|--------|--------|
| 1 | SEIR step (vectorized + proximity S→E) | `hazards.SEIREngine` |
| 2 | Plume concentrations | `hazards.compute_plume_concentration` |
| 3–4 | Wearable observe/detect → tokens | `agents.CitizenAgent` |
| 5 | Eclipse filter + Sybil/replay injection | `attacks.AttackOrchestrator` |
| 6 | Aggregator threshold → dilated broadcast | `agents.NetworkAggregator` |
| 7 | Zone-indexed agent responses | `wearable_agents_by_cell` |
| 8 | Classify detections (zone-local for plume/cardiac) | `simulation._classify_detection` |
| 9 | Deanonymization attack attempt | `_run_deanon_attack` |
| 10 | Episode metrics + step record | `metrics.MetricsCollector` |

## State Storage

| State | Storage | Notes |
|-------|---------|-------|
| Positions | `agent_x`, `agent_y` | **Static** after init (mobility: issue #29) |
| Cell IDs | `agent_cell_ids`, `CitizenAgent.cell_id` | Cached at init; grid cell namespace |
| SEIR | `seir.states` arrays | Vectorized; `max_infectious_checks=500` cap |
| Wearables | `has_wearable`, household/neighborhood IDs | Households nested in neighborhoods |
| Zone index | `wearable_agents_by_cell` | Fast broadcast response lookup |
| Privacy | `aggregator.state` | Token counts, linear ε sum |

**Mesa:** `GarlandModel(mesa.Model)` — organizational only; no Mesa scheduler.

## Spatial Indexing

All privacy protocol paths use **grid cell IDs** (0 … `n_cells-1`):

- Token `zone_id` = `cell_id`
- Dilated zones = list of cell IDs
- Query matching = `cell_id in query.zone_cells`

Neighborhood IDs are for population layout only.

## Attack Layer (all wired)

| Attack | CLI flag | Simulation hook |
|--------|----------|-----------------|
| Sybil | `--enable-sybil` | `step_injections` every 6 steps |
| Deanonymization | `--enable-deanon` | `_run_deanon_attack` periodic |
| Correlation | `--enable-correlation` | `observe_protocol_responses` + `evaluate_periodic` |
| Eclipse | `--enable-eclipse` | `filter_tokens` before aggregation |
| Replay | `--enable-replay` | `cache_tokens_for_replay` + injection |

Metrics synced via `metrics.sync_attack_metrics(orchestrator)`.

## Detection Classification

| Anomaly type | TP ground truth |
|--------------|-----------------|
| RESPIRATORY | Zone-local plume exposure |
| CARDIAC | Plume exposure → toxin; else zone-local disease |
| FEBRILE, MULTI_SYSTEM | **Global** infectious count today (**bug #25**) |

## Performance

- Vectorized init and SEIR/plume for all N agents
- Per-step cost scales with wearables W ≈ 0.15N and broadcast fan-out
- Benchmark: `python -m garland.benchmark --quick`
- See `docs/SCALING.md` for 250K estimates

## Key Config

```
SimulationConfig
├── seir: SEIRConfig (incl. max_infectious_checks)
├── plume: PlumeConfig
├── privacy: PrivacyConfig
└── attacks: AttackConfig (all attack params)
```

## Open architecture issues

- **#25** — zone-local febrile classification
- **#24** — adaptive composition in metrics
- **#29** — agent mobility
- **#32** — H3 indexing

See `../garland-issues/references/known-issues.md`.

## Related Skills

- `garland-privacy-protocol` — protocol details
- `garland-testing` — test layout
- `garland-issues` — backlog
