# Operational detection measurements

GARLAND's episode-level FPR/FNR metrics answer whether an episode was
detected, but not the alert burden experienced by an operator. The committed
scenarios make that burden reproducible.

## Sequential per-person detection

The default `detector_mode: instant` preserves the fixed Mahalanobis gate used by
the earlier operating-point measurements.  Runs may opt into sequential
per-person detection with `detector_mode: sequential`.  Each wearable maintains
a CUSUM of its Mahalanobis distance, with configurable reference value,
threshold, clearing hysteresis, clear-level fraction, and residual EWMA
classification. Sequential state is reset during baseline warm-up and remains
unarmed until the baseline covariance warm-up is complete, so cold-start
adaptation cannot create a latched alert. While an episode remains active,
classified observations continue to emit tokens at the normal wearable
cadence; clearing below the re-arm level for the configured number of steps
ends the episode and permits a later independent alarm.

For example:

```bash
garland --config examples/sequential_onset.yaml --no-plots
```

The summary records the selected detector mode and parameters so instant and
sequential operating curves remain directly auditable.

## Attributed versus coincidental detections

The pre-existing detection counts and latency fields are zone-local metrics:
they count a genuine response in a zone containing an active hazard, even when
the threshold-crossing tokens came from unaffected background agents. This is
intentional historical behavior and those fields remain unchanged for
regression comparability.

Summaries also expose provenance-only causal measurements under flat
`attributed_*`, `coincidental_*`, and `affected_agent_*` keys. A detection is
attributed when at least one affected agent emitted a token in the same
zone/anomaly-type group that crossed the aggregation threshold. Otherwise a
zone-local true positive is counted as coincidental. Attributed latency is
`None` when no attributed detection exists; it is never substituted with the
coincidental latency. The summary reports the coincidental fraction and
separate attributed/coincidental counts for disease and toxin.

Affected-token fragmentation is reported under
`affected_agent_token_counts` and `largest_affected_agent_group`. These fields
count only model-side provenance recorded at token emission: toxin status uses the
existing concentration gate (`> 0.01`), and disease status uses the agent's
SEIR state at emission. The provenance is not part of `EncryptedToken`, is not
available to the aggregator, and cannot affect detection, privacy responses, or
query behavior. The fragmentation breakdown shows whether affected tokens
split across anomaly types or fail to form a same-zone/type group large enough
to reach `threshold_m`.

## Null-baseline methodology

Run `examples/null_baseline.yaml` with no infection and no plumes:

```bash
garland --config examples/null_baseline.yaml --no-plots \
  --output-dir output/null_baseline
```

The resulting `summary.json` contains a per-day `operational_metrics_daily`
series with broadcasts per occupied zone per day, broadcasts per 1,000 agents
per day, and the fraction of occupied zones alarming at least once. It also
contains issued-broadcast precision and epsilon per agent per day. Since the
scenario has no hazards, every alert is a false alarm.

## Undefined metrics

Metrics return `None` when their evidence or denominator is absent: for
example, when no hazard onset occurred, no detection was observed, or a
comparison class was not represented. `None` means undefined; it is not a
zero, one, or an imputed latency.

## Committed scenario-scoped measurements

The figures in this section are measurements of this simulation under the
listed scenario and seed. They are not general performance claims about
wearable surveillance. Each included record is traceable to the artifact
metadata described below; no new simulation was run to produce this record.

### Instant staged attribution

Artifact: `garland_scratch/attributed_instant_staged_final/summary.json`

- **Configuration:** 10,000 agents, 1,728 steps (6 days), seed `42`,
  `detector_mode: instant`, `anomaly_threshold: 3.5`,
  `privacy.threshold_m: 5`, `k_min: 10`, and `time_window_steps: 12`.
  The artifact records plume onset at step 864 and outbreak onset at step
  1152.
- **Coincidental detections:** 95.51% of disease zone-local true positives
  (553 coincidental versus 26 attributed) and 78.63% of toxin zone-local true
  positives (103 coincidental versus 28 attributed).
- **Affected-token fragmentation:** toxin affected-agent tokens were
  4 `RESPIRATORY`, 9 `CARDIAC`, and 1 `MULTI_SYSTEM`. The largest affected
  same-zone/same-type group was 2, below `threshold_m = 5`; the affected
  stream therefore did not independently cross the aggregation threshold in
  this scenario.

These are provenance-only measurements. The historical zone-local counts
remain separate and include detections whose threshold-crossing support came
from unaffected background agents.

### Sequential toxin latency decomposition

Artifact: `garland_scratch/toxin_timeline_seq_h5/summary.json`

- **Configuration:** 10,000 agents, 1,728 response rounds (6 days), seed `42`,
  `detector_mode: sequential`, `anomaly_threshold: 3.5`,
  `sequential_reference_value: 2.0`, `sequential_threshold: 5.0`,
  `sequential_clear_steps: 3`, `sequential_clear_fraction: 0.5`,
  `sequential_residual_ewma_alpha: 0.2`, `threshold_m: 5`, `k_min: 10`,
  and `time_window_steps: 12`.
- The artifact's toxin timeline is 9.25 hours from exposure (step 864) to the
  correct toxin classification (step 975).
- Approximately 2.9 hours elapse from exposure to the first alarm marker
  (step 899), 5.8 hours from that alarm to the first exposed token marker
  (step 969), and 0.5 hours from the exposed-token marker to correct
  classification (step 975). In this scenario, the detector/classification
  path owns the latency; the aggregation path is comparatively small.

### Traceability limitation for null-baseline figures

The retained 30-day null artifacts
`garland_scratch/pr2_null_post_final/`,
`garland_scratch/pr3_final_null_30d_h5/`, and
`garland_scratch/pr3_seq_null_30d_h5/` record population, duration, and
detector parameters, but their `summary.json` files do not record the seed.
The pre-#71 trajectory artifact likewise lacks sufficient scenario metadata.
Their numeric operating points are therefore intentionally not committed
here: a figure without a seed/configuration trace would violate the
reproducibility rule for this record. The two sequential null artifacts also
disagree; the `pr3_final_null_30d_h5` artifact has the merged hysteresis
metadata, but its missing seed still prevents committing either number.

## Operating curve

The staged scenario has a 576-step (two-day) warm-in, a plume onset at step
864, and a disease onset at step 1152:

```bash
garland --config examples/staged_onset.yaml --no-plots \
  --output-dir output/staged_onset
```

Sweep the anomaly threshold directly from the CLI:

```bash
garland sweep --sweep-config examples/operational_detection_sweep.yaml \
  --write-run-outputs
```

The sweep writes `sweep_results.csv`. The `anomaly_threshold` axis is
configuration-wired, while `--anomaly-threshold` provides the equivalent
single-run override. The existing `baseline_decay_lambda` and
`baseline_seasonal_decay` fields are also configurable and retain their
historical defaults.

The current detector fixes two calibration defects without retuning the
threshold: cyclical profiles are learned as EMA-relative deviations, and
covariance is estimated from the same pre-update residual used by the
Mahalanobis score. The default null run therefore remains a deliberately
high-background operating point, but its false-alarm rate should be
stationary rather than diverging over a month.

Plume exposure uses the existing concentration gate of `> 0.01`. Exposed
plume observations are classified as respiratory before the generic
multi-system fallback when they are fever-free; late-stage infection remains
febrile or multi-system because it includes a temperature increase.
