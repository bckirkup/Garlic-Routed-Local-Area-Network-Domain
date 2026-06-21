---
name: garland-privacy-protocol
description: Implement and fix the GARLAND decentralized privacy protocol including tokens, K-anonymity dilution, broadcast queries, and DP responses. Use when changing privacy.py, spatial.py, agents.py aggregator logic, or zone matching behavior.
paths:
  - "src/garland/privacy.py"
  - "src/garland/spatial.py"
  - "src/garland/agents.py"
  - "src/garland/simulation.py"
  - "tests/test_privacy.py"
  - "tests/test_simulation.py"
---

# GARLAND Privacy Protocol

## When to Use

- Fixing or extending the broadcast-and-filter DP framework
- Debugging broadcasts, dilution, or agent responses
- Working on issues #24, #25, or spatial/privacy bugs

## Protocol Stages

### 1. Token emission

`CitizenAgent.observe_and_detect(..., cell_id=...)` → `EncryptedToken(zone_id=cell_id, ...)`

Dummy traffic uses same `cell_id`.

### 2. Threshold aggregation

`AggregatorState.check_thresholds()` — count per `(zone_id, anomaly_type)` in time window; dummies filtered.

### 3. K-anonymity dilution

`grid.dilated_zone(trigger_cell, k_min)` → list of cell IDs.

### 4. Broadcast + response

`wearable_agents_by_cell[cell_id]` iterated for each cell in `query.zone_cells`.

Responses: randomized response + planar Laplace noise on location.

## Spatial Namespace (fixed on main)

All protocol paths use **grid cell IDs**. Neighborhood IDs are layout-only.

Do not regress closed issue #5.

## Privacy Parameters (`PrivacyConfig`)

| Param | Default | Role |
|-------|---------|------|
| `threshold_m` | 5 | Min tokens to broadcast |
| `k_min` | 50 | Min population in dilated zone |
| `time_window_steps` | 12 | 1-hour aggregation window |
| `epsilon_per_response` | 0.1 | Per genuine response |
| `randomized_response_p` | 0.75 | Truthful RR probability |
| `laplace_scale` | 200 m | Planar Laplace scale |
| `dummy_rate` | 0.01 | Dummy packet rate |

## Epsilon Accounting (open bug #24)

Runtime: linear sum `genuine_responses × epsilon_per_response`.

README claims adaptive composition via `compute_adaptive_composition_epsilon()` — function tested but **not used in summary**.

Fix options:
- Wire composition into `MetricsCollector.summary()`
- Or update README/plot labels to "linear ε expenditure"

## Detection Classification (open bug #25)

| Anomaly | TP logic |
|---------|----------|
| RESPIRATORY | `_zone_has_plume_exposure()` ✓ |
| CARDIAC | Plume → toxin; else `_zone_has_active_disease()` ✓ |
| FEBRILE, MULTI_SYSTEM | Global infectious count ✗ → should use zone-local |

## Attack Interactions

| Attack | Privacy impact |
|--------|----------------|
| Eclipse | Drops tokens before aggregation |
| Sybil | Injects fake tokens → false broadcasts |
| Replay | Re-injects cached tokens with new time bin |
| Deanonymization | Narrow single-cell query bypasses dilution (attack test) |
| Correlation | Collects perturbed responses across broadcasts |

## Simulated Crypto

Tokens are plaintext tuples. `agent_id_hash = hash(idx)`. No real homomorphic encryption (#35 documents this).

## Testing

```bash
python3 -m pytest tests/test_privacy.py tests/test_simulation.py::TestProtocolSimulationIntegration -v
python3 -m pytest tests/test_simulation.py::TestDetectionClassification -v
```

Required for protocol changes:
- Unit tests for changed mechanism
- Integration test if matching/dilution logic changes

## Open Issues

| Issue | Type | Topic |
|-------|------|-------|
| #25 | Bug | Zone-local febrile classification |
| #24 | Bug | Adaptive composition in metrics |
| #35 | Doc | Privacy guarantees as design intent |
| #32 | Feature | H3 spatial indexing |

## Related Skills

- `garland-architecture` — full step pipeline
- `garland-testing` — fixtures and CI
- `garland-issues` — backlog
