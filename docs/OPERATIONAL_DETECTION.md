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

Summaries also expose provenance-only causal measurements under
`attributed_detection`. A detection is attributed when at least one affected
agent emitted a token in the same zone/anomaly-type group that crossed the
aggregation threshold. Otherwise a zone-local true positive is counted as
coincidental. Attributed latency is `None` when no attributed detection exists;
it is never substituted with the coincidental latency. The summary also reports
the coincidental fraction and separate attributed/coincidental counts for
disease and toxin.

Affected-token fragmentation is reported separately under
`affected_agent_tokens` and `largest_affected_agent_group`. These fields count
only model-side provenance recorded at token emission: toxin status uses the
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
