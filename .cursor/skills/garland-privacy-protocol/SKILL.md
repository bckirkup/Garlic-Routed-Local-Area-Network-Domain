---
name: garland-privacy-protocol
description: Implement and fix the GARLAND decentralized privacy protocol including tokens, K-anonymity dilution, broadcast queries, and DP responses. Use when changing privacy.py, spatial.py, agents.py aggregator logic, or zone matching behavior.
paths:
  - "src/garland/privacy.py"
  - "src/garland/spatial.py"
  - "src/garland/agents.py"
  - "src/garland/simulation.py"
  - "tests/test_privacy.py"
---

# GARLAND Privacy Protocol

## When to Use

- Fixing or extending the broadcast-and-filter DP framework
- Working on spatial dilution, token aggregation, or agent responses
- Debugging why broadcasts miss agents or target wrong zones
- Implementing issue #5 (zone ID mismatch)

## Protocol Stages

### 1. Blind gating (token emission)

`CitizenAgent.observe_and_detect()` emits `EncryptedToken` when Mahalanobis distance > 3.5:

```python
EncryptedToken(
    zone_id=...,           # MUST be grid cell ID (see #5)
    anomaly_type=...,
    timestamp_bin=...,
    agent_id_hash=...,
    is_dummy=False,
)
```

Dummy traffic: `generate_dummy_traffic()` and non-matching RR responses.

### 2. Threshold aggregation

`AggregatorState.receive_token()` — dummies filtered out.

`check_thresholds(time_bin, config)` — counts tokens per `(zone_id, anomaly_type)` within `time_window_steps` (default 12 = 1 hour). Triggers when count ≥ `threshold_m` (default 5).

### 3. K-anonymity spatial dilution

`NetworkAggregator.evaluate_and_broadcast()` calls:

```python
dilated_cells = spatial_dilate_fn(zone_id, config.k_min)
```

`SpatialGrid.dilated_zone(center_cell, k_min)` expands rings until population ≥ `k_min`.

**Bug (#5):** `zone_id` from tokens is currently `neighborhood_id`, not `center_cell`. Fix:

```python
# In CitizenAgent — pass cell_id from simulation context
zone_id=cell_id  # grid.cell_of(self.idx)

# In respond_to_query — match on cell_id
cell_id in query.zone_cells  # not neighborhood_id
```

### 4. Reverse-query broadcast

`BroadcastQuery` contains:

- `zone_cells` — list of dilated grid cell IDs
- `anomaly_type` — agents check match against active anomaly
- `time_window_start/end`

### 5. Uplink perturbation

`CitizenAgent.respond_to_query()`:

1. `randomized_response(matches, p=0.75)` — coin-flip DP on anomaly match
2. If reporting match: `planar_laplace_noise(laplace_scale)` added to `(x, y)`
3. If non-match: optional dummy with larger noise (`scale * 2`)
4. Increment `local_epsilon` by `epsilon_per_response`

## Privacy Parameters (`PrivacyConfig`)

| Param | Default | Meaning |
|-------|---------|---------|
| `threshold_m` | 5 | Min tokens to trigger broadcast |
| `k_min` | 50 | Min population in dilated zone |
| `time_window_steps` | 12 | Aggregation window (1 hour) |
| `epsilon_per_response` | 0.1 | Budget per genuine response |
| `randomized_response_p` | 0.75 | Truthful RR probability |
| `laplace_scale` | 200 m | Planar Laplace scale |
| `dummy_rate` | 0.01 | Dummy packet emission rate |

## Anomaly Types

```python
class AnomalyType(Enum):
    RESPIRATORY = "respiratory"   # Toxin-primary (no fever)
    CARDIAC = "cardiac"
    FEBRILE = "febrile"           # Infection-primary
    MULTI_SYSTEM = "multi_system"
```

Classification: `classify_anomaly(obs, baseline)` in `privacy.py`.

## Spatial Grid Reference

```python
grid = SpatialGrid(width, height, cell_size=200.0)
grid.assign_positions(agent_x, agent_y)
cell_id = grid.cell_of(agent_idx)          # 0 .. n_cells-1
zone = grid.dilated_zone(cell_id, k_min)   # list of cell IDs
pop = grid.zone_population(cell_id)
```

Cell ID formula: `row * cols + col` where row/col from position / cell_size.

**Do not mix** neighborhood IDs into this pipeline.

## Homomorphic Encryption

Production would use Paillier/BFV. This testbed **simulates** protocol semantics only — tokens are plaintext tuples. Do not implement real crypto unless explicitly requested.

## Privacy Guarantees (Design Intent)

README claims (not formally verified in code):

1. No single query unmasks exact location (Planar Laplace + K-anonymity)
2. Dilated zones contain ≥ K agents
3. Adaptive composition bounds total ε
4. Randomized response provides plausible deniability

Treat these as **design goals**; validate with tests, not assertions in docs alone.

## Testing Privacy Changes

Required after any protocol change:

```bash
python3 -m pytest tests/test_privacy.py -v
python3 -m pytest tests/test_simulation.py::TestEndToEnd -v
```

Add integration test (issue #12):

1. Force N wearable agents in same cell to emit same `anomaly_type` tokens
2. Step until broadcast triggers
3. Assert dilated zone includes source cell
4. Assert responding agents have `cell_id in query.zone_cells`
5. Assert `responses_received > 0`

See `garland-testing` skill for fixture patterns.

## Common Pitfalls

| Pitfall | Consequence |
|---------|-------------|
| Using `neighborhood_id` as `zone_id` | Dilution and matching break (#5) |
| Testing dilution with cell IDs but production with neighborhood IDs | Tests pass, simulation fails |
| Counting dummy tokens in threshold | Prevented — aggregator filters `is_dummy` |
| Global plume timing for detection TP | Misleading metrics (#2) |

## Related Issues

- **#5** — Zone ID mismatch (fix first)
- **#2** — Detection classification
- **#7** — Metrics not wired
- **#12** — Missing integration tests

## Related Skills

- `garland-architecture` — full step pipeline
- `garland-testing` — test conventions
- `garland-issues` — backlog and PR workflow
