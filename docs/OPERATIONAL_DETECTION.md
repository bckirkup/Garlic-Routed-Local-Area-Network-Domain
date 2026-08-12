# Operational detection measurements

GARLAND's episode-level FPR/FNR metrics answer whether an episode was
detected, but not the alert burden experienced by an operator. The committed
scenarios make that burden reproducible.

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
