---
name: garland-privacy-protocol
description: Work on GARLAND privacy protocol — tokens, K-anonymity dilution, broadcasts, DP responses, and detection classification. Use when editing privacy.py, spatial.py, agents.py, or simulation classification logic.
paths:
  - "src/garland/privacy.py"
  - "src/garland/spatial.py"
  - "src/garland/agents.py"
  - "src/garland/simulation.py"
  - "tests/test_privacy.py"
  - "tests/test_simulation.py"
---

# GARLAND Privacy Protocol

## Protocol Overview

Broadcast-and-filter decentralized DP testbed (simulated homomorphic encryption):

```
observe → EncryptedToken[cell_id, anomaly_type]
       → aggregator threshold (count in time window)
       → dilated_zone(cell_id, k_min)
       → BroadcastQuery[zone_cells, anomaly_type]
       → randomized_response + planar_laplace(location)
       → MetricsCollector
```

## Key Types

```python
EncryptedToken(zone_id, anomaly_type, timestamp_bin, agent_id_hash, is_dummy)
BroadcastQuery(zone_cells, anomaly_type, time_window_start/end, query_id)
PerturbedResponse(query_id, reported_x/y, anomaly_confirmed, is_dummy)
```

## Mechanisms (`privacy.py`)

| Function | Role |
|----------|------|
| `planar_laplace_noise(scale, rng)` | Geo-indistinguishability on (x, y) |
| `randomized_response(truth, p, rng)` | Plausible deniability on anomaly match |
| `compute_adaptive_composition_epsilon(n, eps, delta)` | Advanced composition bound |
| `classify_anomaly(obs, baseline)` | RESPIRATORY / FEBRILE / CARDIAC / MULTI_SYSTEM |

## Parameters (`PrivacyConfig`)

| Param | Default | Meaning |
|-------|---------|---------|
| `threshold_m` | 5 | Tokens to trigger broadcast |
| `k_min` | 50 | Min population in dilated zone |
| `time_window_steps` | 12 | 1-hour aggregation window |
| `epsilon_per_response` | 0.1 | Budget per genuine response |
| `randomized_response_p` | 0.75 | Truthful RR probability |
| `laplace_scale` | 200 m | Location noise scale |
| `dummy_rate` | 0.01 | Dummy packet emission rate |

## Spatial Rules

- Token `zone_id` = agent **grid cell ID**
- Query matching: `cell_id in query.zone_cells`
- Response iteration: `wearable_agents_by_cell[cell_id]`
- Dummies filtered at aggregator — do not count toward threshold

## Epsilon Reporting

Summary tracks cumulative ε from genuine responses. Adaptive composition function exists for theoretical bounds — ensure README/summary wording matches whichever accounting model is implemented.

## Detection ↔ Hazard Mapping

| Hazard signal | Biometric pattern | Anomaly type |
|---------------|-------------------|--------------|
| Infection | Fever + HR↑ + RR↑ | FEBRILE, MULTI_SYSTEM |
| Toxin | RR↑, no fever | RESPIRATORY |
| Stress/isolated | HR/HRV | CARDIAC |

Classification TP/FP must use **zone-local** exposure/disease checks — see `TestDetectionClassification`.

## Attack Interactions

| Attack | Effect on protocol |
|--------|-------------------|
| Eclipse | Removes tokens before aggregation |
| Sybil | Floods zone with fake tokens → spurious broadcasts |
| Replay | Re-submits aged tokens with current time bin |
| Deanonymization | Single-cell query bypasses dilution (adversarial test) |
| Correlation | Links perturbed responses across time |

## Simulated vs Production

- Tokens are **plaintext tuples** (not encrypted)
- `agent_id_hash = hash(idx)` — not cryptographic
- README privacy guarantees are **design goals**, not formal proofs

## Tests to Run

```bash
python3 -m pytest tests/test_privacy.py -v
python3 -m pytest tests/test_simulation.py::TestDetectionClassification -v
python3 -m pytest tests/test_simulation.py::TestProtocolSimulationIntegration -v
```

## Change Checklist

- [ ] Cell ID namespace consistent end-to-end
- [ ] Zone-local classification for all anomaly types
- [ ] Regression test added
- [ ] Dummy tokens still filtered at aggregator

## References

- `../garland-architecture/SKILL.md`
- `../garland-testing/SKILL.md`
- `../garland-issues/references/resolved-issues.md`
