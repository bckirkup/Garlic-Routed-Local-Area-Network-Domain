---
name: garland-architecture
description: Understand GARLAND simulation architecture, modules, and data flow. Use when navigating the codebase, planning changes, or debugging the step loop.
paths:
  - "src/garland/**"
  - "README.md"
  - "docs/**"
---

# GARLAND Architecture

## System Purpose

Simulate a city (default **250,000** agents) at **5-minute** steps with:

1. Co-occurring **SEIR disease** + **Gaussian plume toxin**
2. **Wearable biometrics** (~15%) with Mahalanobis anomaly detection
3. **Privacy protocol** — tokens → threshold → K-anonymity dilution → broadcast → DP responses
4. **Attack evaluation** — five adversarial models with summary metrics

## Data Flow

```
app.py (CLI)
    └── GarlandModel.step()  [simulation.py]
            ├── SEIREngine.step()           [hazards]
            ├── compute_plume_concentration()
            ├── CitizenAgent.observe_and_detect()  → tokens
            ├── AttackOrchestrator             → filter/inject
            ├── NetworkAggregator              → broadcast queries
            ├── CitizenAgent.respond_to_query()  → perturbed responses
            ├── _classify_detection()          → DetectionEvent
            ├── _run_deanon_attack()
            └── MetricsCollector               → episode FPR/FNR
```

## Step Pipeline (one 5-min tick)

| # | Stage | Key code |
|---|-------|----------|
| 1 | SEIR transitions | `seir.step()` — vectorized E→I, I→R; proximity S→E (capped) |
| 2 | Plume | `compute_plume_concentration()` — all agents |
| 3 | Biometrics + tokens | loop `citizen_agents` — only wearables |
| 4 | Attacks | `filter_tokens()` (eclipse), `step_injections()` (sybil/replay) |
| 5 | Aggregate + broadcast | `evaluate_and_broadcast()` + `dilated_zone()` |
| 6 | Responses | `wearable_agents_by_cell[cell_id]` per query zone |
| 7 | Classify + deanon | `_classify_detection`, `_run_deanon_attack` |
| 8 | Metrics | `update_hazard_episode`, `record_step` |

## State Model

| State | Storage | Notes |
|-------|---------|-------|
| Positions | `agent_x`, `agent_y` | Static after init |
| Cell membership | `agent_cell_ids`, `CitizenAgent.cell_id` | Grid cell namespace |
| Wearable index | `wearable_agents_by_cell` | O(wearables in zone) lookups |
| SEIR | `seir.states` | `max_infectious_checks=500` at scale |
| Households | nested in neighborhoods | Patchy wearable assignment |
| Privacy | `aggregator.state` | Token counts, ε tracking |

## Spatial Index

- Rectangular grid: `cell_size` default 200 m
- **All privacy zones use grid cell IDs** (0 … `n_cells−1`)
- `dilated_zone(center_cell, k_min)` expands rings until population ≥ K
- Neighborhood IDs are for layout only — not used in protocol

## Detection Classification

Uses **zone-local ground truth** where implemented:

| Anomaly | TP condition |
|---------|--------------|
| RESPIRATORY | `_zone_has_plume_exposure(zone_cells)` |
| CARDIAC | Plume in zone → toxin; else `_zone_has_active_disease(zone_cells)` |
| FEBRILE, MULTI_SYSTEM | Zone-local disease (see `TestDetectionClassification`) |

## Attack Layer

| Attack | CLI | Hook |
|--------|-----|------|
| Sybil | `--enable-sybil` | Inject fake tokens every 6 steps |
| Deanonymization | `--enable-deanon` | Periodic narrow query on target cell |
| Correlation | `--enable-correlation` | Observe responses; periodic distinguish test |
| Eclipse | `--enable-eclipse` | Drop tokens in target zones pre-aggregation |
| Replay | `--enable-replay` | Re-inject cached stale tokens |

Metrics synced: `metrics.sync_attack_metrics(orchestrator)`.

## Performance

- Vectorized init, SEIR, plume for all N
- Per-step cost ∝ wearables W + broadcast fan-out
- See `docs/SCALING.md` and `python -m garland.benchmark`

## Config Hierarchy

```
SimulationConfig
├── seir: SEIRConfig
├── plume: PlumeConfig
├── privacy: PrivacyConfig
└── attacks: AttackConfig
```

## Mesa

`GarlandModel(mesa.Model)` — naming only; no Mesa scheduler or agent space.

## Related Skills

- `garland-privacy-protocol` — protocol detail
- `garland-testing` — test map
- `garland-issues/references/resolved-issues.md` — regression list
