---
name: garland-architecture
description: Understand GARLAND simulation architecture, data flow, and module boundaries. Use when navigating the codebase, planning changes, debugging the step loop, or explaining how hazards, biometrics, and privacy interact.
paths:
  - "src/garland/**"
  - "README.md"
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
2. **Wearable biometrics** — subset of agents (~15%) with anomaly detection
3. **Privacy protocol** — blind gating → threshold aggregation → K-anonymity dilution → broadcast → DP responses
4. **Attack evaluation** — adversarial models (partially implemented)

## Layer Model

```
┌─────────────────────────────────────────────────────────┐
│  CLI (app.py)                                           │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  GarlandModel (simulation.py)                           │
│  Flat NumPy arrays: positions, SEIR state, wearables    │
└──┬──────────┬──────────┬──────────┬──────────┬─────────┘
   ▼          ▼          ▼          ▼          ▼
 hazards  biometrics  spatial   privacy   attacks
 (SEIR +   (profiles,  (cell     (tokens,  (Sybil,
  plume)    baselines)  grid)     DP)       deanon…)
                           │
                           ▼
                      metrics (CSV, plots, summary)
```

## Simulation Step Pipeline

Each `GarlandModel.step()` (5 minutes simulated):

| Step | Action | Module |
|------|--------|--------|
| 1 | Advance SEIR (E→I, I→R, S→E via proximity) | `hazards.SEIREngine` |
| 2 | Compute plume concentration per agent | `hazards.compute_plume_concentration` |
| 3–4 | Wearable agents: observe biometrics, detect anomalies, emit tokens | `agents.CitizenAgent`, `biometrics` |
| 5 | Inject Sybil tokens (if enabled) | `attacks.AttackOrchestrator` |
| 6 | Aggregator ingests tokens, checks threshold, dilates zone, broadcasts | `agents.NetworkAggregator`, `spatial.SpatialGrid` |
| 7 | Agents in zone respond with randomized response + Laplace noise | `agents.CitizenAgent.respond_to_query` |
| 8 | Classify detections, record missed detections | `simulation._classify_detection`, `metrics` |
| 9 | Record per-step metrics | `metrics.MetricsCollector` |

## State Storage

| State | Storage | Notes |
|-------|---------|-------|
| Agent positions | `agent_x`, `agent_y` (float32 arrays) | Fixed after init — no mobility |
| SEIR | `seir.states`, exposure/infection steps | Vectorized |
| Wearables | `has_wearable` bool array | Patchy by household |
| Neighborhood | `neighborhood_ids` | Random assignment to cluster centers |
| Household | `household_ids` | Sequential blocks (see issue #9) |
| Biometric profiles | `profiles` list | One per wearable agent |
| Privacy | `aggregator.state` | Token counts, epsilon budget |

**Mesa:** `GarlandModel` extends `mesa.Model` but does not use Mesa scheduler or agent space — organizational choice only.

## Spatial Indexing (Critical)

Two ID systems exist today (bug — issue #5):

| ID type | Range | Used for |
|---------|-------|----------|
| **Grid cell ID** | 0 … `n_cells-1` | `SpatialGrid`, dilated zones, population counts |
| **Neighborhood ID** | 0 … `n_neighborhoods-1` | Position clustering, token `zone_id` (incorrect) |

**Correct model:** tokens, dilution, and query matching should all use **grid cell IDs** from `grid.cell_of(agent_idx)`.

## Privacy Protocol Flow

```
CitizenAgent.observe_and_detect()
    → EncryptedToken(zone_id, anomaly_type, timestamp_bin)
    → NetworkAggregator.ingest_tokens()
    → check_thresholds() → (zone, anomaly_type) triggers
    → spatial_dilate_fn(zone, k_min) → BroadcastQuery(zone_cells, ...)
    → CitizenAgent.respond_to_query() in matching zone
    → PerturbedResponse(reported_x/y, anomaly_confirmed)
    → MetricsCollector
```

**DP mechanisms** (`privacy.py`):

- `planar_laplace_noise` — geo-indistinguishability on location
- `randomized_response` — plausible deniability on anomaly match
- `compute_adaptive_composition_epsilon` — budget accounting (documented, not fully wired to per-query tracking)

## Hazard → Biometric Mapping

| Hazard | Biometric signature | Anomaly type |
|--------|---------------------|--------------|
| Infection (I) | Fever + HR↑ + RR↑ + HRV↓ | `FEBRILE`, `MULTI_SYSTEM` |
| Toxin (plume) | RR↑ + HR↑ + HRV↓, **no fever** | `RESPIRATORY` |
| Incubation (E) | Subtle HRV↓ | May not trigger threshold |

Classification logic: `privacy.classify_anomaly()` from deviation pattern.

## Attack Layer (Current)

| Class | Status in simulation |
|-------|---------------------|
| `SybilAttacker` | Injected every 6 steps via orchestrator |
| `DeanonymizationAttacker` | Class only — not in step loop |
| `CorrelationAttacker` | Class only |
| `EclipseAttacker` | Class only |
| `MaliciousAgent` | Constructed but unused |

## Performance Design

- Vectorized SEIR and plume for all agents
- Python loop over `citizen_agents` only (~15% of n)
- SEIR transmission samples max 500 infectious agents per step
- Position init uses Python loop over all n (bottleneck at 250K)

## Key Config Objects

```
SimulationConfig
├── seir: SEIRConfig
├── plume: PlumeConfig
├── privacy: PrivacyConfig
└── attacks: AttackConfig
```

Defaults in `simulation.py` and overridable via CLI (`app.py`).

## Related Skills

- `garland-privacy-protocol` — detailed privacy/spatial fix guidance
- `garland-testing` — what to test per layer
- `garland-issues` — known bugs by module
