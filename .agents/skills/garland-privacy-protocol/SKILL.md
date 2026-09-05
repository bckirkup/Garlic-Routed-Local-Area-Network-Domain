---
name: garland-privacy-protocol
description: Work on GARLAND privacy protocol — tokens, K-anonymity, broadcasts, DP responses, zone-local detection. Use when editing privacy, spatial, agents, or simulation classification.
paths:
  - "src/garland/privacy.py"
  - "src/garland/spatial.py"
  - "src/garland/agents.py"
  - "src/garland/simulation.py"
  - "tests/test_privacy.py"
  - "tests/test_simulation.py"
  - "tests/test_spatial.py"
---

# GARLAND Privacy Protocol

## Flow

```
observe(cell_id) → EncryptedToken
  → aggregator threshold
  → dilated_zone(cell_id, k_min)   # H3 or rect ring expansion
  → BroadcastQuery[zone_cells]
  → randomized_response + planar_laplace
  → zone-local classification
  → MetricsCollector

At token emission, the model also records affected-agent provenance for
measurement, including a cause set and disease/toxin hazard booleans. This is
a separate oracle: provenance is deliberately not part of `EncryptedToken` or
`BroadcastQuery`/`PerturbedResponse`, and is unavailable to aggregation,
dilation, responses, queries, or classification.
```

## Rules

- **Cell IDs** from active `SpatialIndex` (H3 or rectangular)
- Dummies filtered at aggregator
- Responses keyed by `wearable_agents_by_cell`
- Tokens simulate encryption (plaintext tuples in testbed)

## Detection Classification (zone-local)

Zone-local TP means that a query landed in a zone containing the relevant
hazard instance. It does **not** mean that the threshold-crossing tokens came
from affected agents. Provenance-only metrics separately label such detections
as attributed or coincidental.

| Anomaly | TP logic |
|---------|----------|
| RESPIRATORY | `_zone_plume_instance(zone_cells)` |
| FEBRILE, MULTI_SYSTEM | `_zone_outbreak_instance(zone_cells)` |
| CARDIAC | Plume instance → toxin; else outbreak instance → disease |

## Parameters (`PrivacyConfig`)

`threshold_m`, `k_min`, `time_window_steps`, `epsilon_per_response`, `randomized_response_p`, `laplace_scale`, `dummy_rate`.

## Epsilon

Summary reports cumulative ε from responses. README frames privacy claims as
**design goals** — verify adaptive composition wording matches implementation
(#24). `_classify_detection` itself never reads `reported_x`/`reported_y`.
By default geo noise changes reported coordinates and accounting only; with
`privacy.geo_zone_filter: true`, `_filter_responses_to_zone` drops replies
whose perturbed position maps (`grid.cell_for_position`) outside the queried
zone before aggregation, so `laplace_scale` costs detection evidence (#75).
Dropped replies are summed in `geo_zone_filtered_responses`; attack
instrumentation still sees the unfiltered broadcast replies.

## Attacks vs Protocol

| Attack | Effect |
|--------|--------|
| Eclipse | Drops tokens pre-aggregation |
| Sybil / Replay | Inflates token counts |
| Deanonymization | Narrow single-cell query (adversarial) |
| Correlation | Links responses over time |

## Tests

```bash
python -m pytest tests/test_privacy.py tests/test_spatial.py -v
python -m pytest tests/test_simulation.py::TestDetectionClassification -v
python -m pytest tests/test_simulation.py::TestProtocolSimulationIntegration -v
```

## Change Checklist

- [ ] Cell ID consistent across mobility updates
- [ ] Both spatial backends if touching dilution
- [ ] Regression test for classification changes

## References

- `garland-architecture`, `garland-testing`
- `resolved-issues.md` (#5, #25)
