# Operational detection measurements

## Second-round disambiguation

The optional disambiguation layer is an interpretation aid, not validation.
After a zone trigger, the aggregator may ask whether a configured hypothesis
such as recent device adoption could explain the cluster. The simulated human
approval is seeded model behavior; no device reports age, adoption step, or
other per-device metadata.

Acknowledgements are automatic and content-free: they indicate only that a
device is reachable in the queried zone. They are released as a noised,
zone-level count subject to the existing `k_min` floor. An ack is separate from
the human answer. A reachable person may approve yes or no, or provide no
answer. Non-response is free, never inferred as a negative, and expires as an
unresolved hypothesis. Both approved answer arms are charged separately from
the round-one response budget. Reported yes/no counts are
randomized-response perturbed rather than raw human answers; an affirmation
count is contextual evidence, not ground truth or validation. These mechanics
are simulation measurements, not a formal DP proof or a claim of real
encryption.

## Benign confounder engine

The disabled-by-default `confounders` sub-config adds model-side,
cause-labelled biometric perturbations for specificity experiments. Exercise,
sleep disruption, and sensor artifacts are independent per-agent sources. The
optional heat-wave source applies a shared ambient excursion across all zones
and records a `heat_<n>` instance with its active-step footprint.
Venue crowding can be enabled for selected venue types and scales its shared
signal by occupancy. Background ILI is an exogenous household process with
configurable incidence, incubation, symptoms, and secondary probability.

Confounder labels are contextual evidence, not validation, and this phase does
not alter hazard classification or scoring. Heat-wave and other cause counts
are model-side metrics; they are not added to encrypted tokens or interpreted
as ground truth by the protocol.

## Benign confounder engine

The disabled-by-default `confounders` sub-config adds model-side,
cause-labelled biometric perturbations for specificity experiments. Exercise,
sleep disruption, and sensor artifacts are independent per-agent sources. The
optional heat-wave source applies a shared ambient excursion across all zones
and records a `heat_<n>` instance with its active-step footprint.
Venue crowding can be enabled for selected venue types and scales its shared
signal by occupancy. Background ILI is an exogenous household process with
configurable incidence, incubation, symptoms, and secondary probability.

Confounder labels are contextual evidence, not validation, and this phase does
not alter hazard classification or scoring. Heat-wave and other cause counts
are model-side metrics; they are not added to encrypted tokens or interpreted
as ground truth by the protocol.

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
wearable surveillance. Each record is anchored on a committed configuration
and an exact invocation; where provenance or timeline instrumentation was
needed, the result was measured in-session rather than generated by the
ordinary CLI summary.

### Null operating point

The committed `examples/null_baseline.yaml` supplies 10,000 agents, seed 42,
anomaly threshold 3.5, static mobility, no infection, and no plumes. The
30-day runs below override its seven-day `n_steps: 2016` value to 8,640
five-minute steps:

```bash
garland --config examples/null_baseline.yaml \
  --n-steps 8640 --no-plots \
  --output-dir output/null_instant_30d
```

The current-main seed-42 run measured **12.89 broadcasts per occupied zone
per day** and **59.3 per 1,000 agents per day**. This daily broadcast count
is seed-sensitive. Two additional current-main runs using the same committed
configuration and duration were:

```bash
garland --config examples/null_baseline.yaml --seed 43 \
  --n-steps 8640 --no-plots \
  --output-dir output/null_instant_30d_seed43

garland --config examples/null_baseline.yaml --seed 44 \
  --n-steps 8640 --no-plots \
  --output-dir output/null_instant_30d_seed44
```

Their results were **10.41 / 51.0** and **11.40 / 53.6** respectively
(broadcasts per occupied zone per day / per 1,000 agents per day). Across
seeds 42–44, the observed range was **10.41–12.89** and **51.0–59.3**.
The earlier 13.83 / 63.6 value is therefore outside this three-seed
current-main sample, but close enough to the observed seed variation that it
should not be treated as a universal operating point.

The more stable quantity is the per-step false-anomaly rate: after the
startup day, these runs remained approximately **0.33–0.46% per operational
wearable per five-minute step**, flat across the month. Daily broadcast
counts are consequently a seed-sensitive summary of that stable background,
not a standalone general performance claim.

For the merged sequential detector, the same committed null configuration was
run with these overrides:

```bash
garland --config examples/null_baseline.yaml \
  --n-steps 8640 --detector-mode sequential \
  --sequential-threshold 5.0 --sequential-clear-steps 3 \
  --no-plots --output-dir output/null_sequential_30d
```

This measured **8.93 broadcasts per occupied zone per day** and **41.1 per
1,000 agents per day**. The run's summary records the merged sequential
parameters, settling this as the merged-implementation result; the older
0.15-background result is not current behavior. The pre-#71 month-long
divergence was a historical defect fixed by #71, not a current operating
point.

### Instant staged attribution

The committed `examples/staged_onset.yaml` supplies 10,000 agents, 1,728
steps (6 days), seed 42, instant mode by default, anomaly threshold 3.5,
plume onset at step 864, outbreak onset at step 1152, and
`privacy.threshold_m: 5`, `k_min: 10`, and `time_window_steps: 12`. The
provenance result was measured in-session with:

```bash
garland --config examples/staged_onset.yaml \
  --detector-mode instant --n-steps 1728 --no-plots \
  --output-dir output/staged_instant_attribution
```

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

The committed `examples/staged_onset.yaml` supplies the six-day, seed-42
staged hazards. The sequential timeline was measured in-session with these
exact detector and protocol overrides:

```bash
garland --config examples/staged_onset.yaml \
  --n-steps 1728 --detector-mode sequential \
  --sequential-reference-value 2.0 --sequential-threshold 5.0 \
  --sequential-clear-steps 3 --sequential-clear-fraction 0.5 \
  --sequential-residual-ewma-alpha 0.2 --no-plots \
  --output-dir output/staged_sequential_threshold5
```

The timeline is 9.25 hours from exposure (step 864) to the correct toxin
classification (step 975). The intermediate marker steps below are not part of
the CLI summary; they were read from additional in-session instrumentation over
the same run, so the invocation above reproduces the run but not the
decomposition.

- Approximately 2.9 hours elapse from exposure to the first alarm marker
  (step 899), 5.8 hours from that alarm to the first exposed token marker
  (step 969), and 0.5 hours from the exposed-token marker to correct
  classification (step 975). In this scenario, the detector/classification
  path owns the latency; the aggregation path is comparatively small.

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

### Background assessment baseline

The background measurement layer records only non-dummy tokens from
operational wearables past their configured baseline warm-up whose model-side
provenance is neither toxin-affected nor disease-affected. It does not add
provenance to protocol tokens or alter aggregation, detection, privacy
responses, or query behavior. Two distinct statistics are reported. The
**emission-level** statistic uses `(zone_id, anomaly_type, timestamp_bin)` and
measures counts accumulated into one emission timestamp bin. The
**aggregation-window** statistic uses `(zone_id, anomaly_type)` at each closed
rolling window and is the operational headline: it matches the token count
that the aggregator evaluates for a broadcast.

The configured `time_window_steps` value is used as a number of timestamp
bins by the existing aggregation check. With the default value 12, one
timestamp bin is one hour and the trigger window spans 12 bins,
approximately 12 hours. There is no per-agent deduplication, and a trigger
clears the queued `(zone_id, anomaly_type)` history except for the current
bin. This section measures those observed implementation semantics; it does
not change the protocol.

Both statistics use heterogeneous-Poisson Pearson dispersion, occupancy
buckets, and observed-versus-Poisson-tail fractions. Groups with zero
expected count are excluded and counted explicitly. If `threshold_m` is not
configured, both tail fractions are undefined (`None`), rather than treating
the threshold as zero.

The assessment has a measurement-only `world_settling_steps` exclusion
setting. Its default is one simulated day derived from the five-minute step
duration (288 steps); setting it to 0 disables exclusion. The deprecated
`background_burn_in_steps` configuration key remains accepted as an alias.
This is separate from `baseline_warmup_steps`, which defaults to 0 and is
absent from `examples/null_baseline.yaml`. The `background_settled_*` fields
use only steps at or after the world-settling boundary, and their rolling
window does not inherit pre-boundary counts.

Every summary and sweep row carries explicit world-settling markers:
`world_settling_steps`, `world_settling_complete`, `world_settling_status`,
`steps_before_world_settling`, `steps_after_world_settling`, and
`world_settling_fraction_of_run`. `world_settling_status` is `not_settled`
when the run ends at or before the boundary and `settled` only when it extends
beyond it. Per-step CSV output carries `past_world_settling`.

Fleet cold start and device onboarding are measured labels, not exclusions.
`fleet_cold_start` identifies a fleet that begins with every wearable in the
existing `BaselineTracker.n_samples < 5` covariance-prior regime and, more
specifically, records whether cold-baseline behavior reached the protocol.
This is a code-defined covariance-prior state, not a baseline-convergence
measure; five samples represent roughly 25 minutes at the default cadence.
`fleet_cold_baseline_wearable_step_fraction` measures the full-run fraction of
wearable-steps in that covariance-prior regime, and
`post_world_settling_cold_baseline_wearable_step_fraction` measures the
post-settling covariance-prior contribution, which can now include genuinely
new adopters. The explicit onboarding-window label is tracked separately from
that detector-state quantity. `device_re_adoption_count` and
`legacy_device_adoption_warmup_reset_count` keep re-adoptions and legacy reset
behavior separate. A retained baseline on device return therefore does not
create a new covariance-prior state. Undefined denominators are `None`, never
zero.
First-time adoption is configured through the `adoption` sub-config:
`all_at_start` is the historical fleet cold start, `rollout` expresses a
deployment ramp, `trickle` expresses individual adopters, and `cohort` adopts
household- or venue-linked groups together. Per-step rows report
`not_adopted_wearables`; summaries include adoption events with step and zone.
`initial_adopted_fraction` leaves an established population in place before
the schedule starts. `onboarding_window_steps` defaults to one simulated day
and labels device age independently of the covariance-prior regime
(`BaselineTracker.n_samples < 5`, about 25 minutes at five-minute steps).
That covariance-prior field is not a baseline-convergence measure. First-time
adopters receive baseline warm-up, while retained baselines on re-adoption do
not. For venue cohorts, `venue_kind` can select a specific venue type;
`any` uses workplace, school, hospital, third place, shopping, sporting,
extended-family, then gathering assignments. The summary field
`peak_onboarding_wearables_in_zone` measures the zone-local onboarding-window
peak independently of covariance-prior status; the existing
`peak_onboarding_cold_wearables_in_zone` remains the narrower intersection.

World settling is the only reporting exclusion. These markers are
observational only and do not alter simulation, detector, aggregation,
privacy, query, or response behavior. Baseline warm-up suppresses dummy
traffic as well as ordinary anomaly tokens, and legacy device re-adoption can
restart local warm-up. Treat every table below as full-run or settled exactly
as labelled, rather than silently treating an unsettled number as an operating
point.

The committed seven-day null baseline remains reproducible with:

```bash
garland --config examples/null_baseline.yaml --n-steps 2016 --no-plots \
  --output-dir output/background_null_baseline
```

With seed 42, 10,000 agents, static hex spatial indexing, and anomaly
threshold 3.5, the earlier seven-day run measured **0.812%** background
tokens per eligible wearable-step and emission-level dispersion **27.4070**,
including the startup transient (the full-run fields).
The emission-level observed and Poisson-tail fractions at `threshold_m = 5`
were **2.886%** and **4.363%**, respectively. These are measurements of that
exact invocation, not a claim that the null model is Poisson.

The rate/threshold curve is reproducible with:

```bash
garland sweep --sweep-config examples/background_assessment_sweep.yaml \
  --output-dir output/background_assessment_sweep --write-run-outputs
```

The committed sweep uses the null baseline and the anomaly-threshold ladder
3.0, 3.5, 4.0, 4.5, and 5.0. Its `sweep_results.csv` includes the background
rate, Pearson dispersion, occupancy-bucket summaries, observed threshold
fraction, and Poisson-tail fraction for each run.

For the seed-42 seven-day sweep, the emission-level curve was:
This run had `world_settling_status: settled` with 288 configured
world-settling steps; the table labels both full-run and settled views
explicitly.

| anomaly threshold | full rate | settled rate | full dispersion | settled dispersion | full observed tail | settled observed tail | full Poisson tail | settled Poisson tail |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.0 | 2.666% | 1.857% | 23.1820 | 3.3129 | 12.626% | 11.772% | 17.981% | 10.622% |
| 3.5 | 0.812% | 0.398% | 27.4070 | 1.5071 | 2.886% | 1.887% | 4.363% | 1.562% |
| 4.0 | 0.356% | 0.063% | 36.7420 | 1.0311 | 0.421% | 0.004% | 1.497% | 0.002% |
| 4.5 | 0.268% | 0.009% | 40.6623 | 1.6462 | 0.178% | 0.000% | 1.065% | 0.0000002% |
| 5.0 | 0.253% | 0.001% | 41.5922 | 1.1318 | 0.149% | 0.000% | 0.981% | 0.000000000006% |

These are scenario- and seed-sensitive measurements, not general claims
about wearable surveillance. The decreasing rate is the detector-threshold
sensitivity; the increasing dispersion reflects the increasingly sparse
high-threshold stream under this baseline.

At threshold 3.5 in the same invocation, the aggregation-window statistic
was **46.2714** full-run dispersion and **2.5218** settled dispersion, with
observed tails **9.388%** and **8.247%**, and Poisson-tail predictions
**31.395%** and **13.275%**, respectively. The window-matched values are the
operational broadcast-burden comparison; the ladder above keeps the
emission-level fields explicit for continuity with the original measurement.

For the mechanism decomposition, the following in-session invocation used the
same seed and configuration shape, with 1,000 agents and one week to keep the
instrumentation inexpensive:

```bash
python /tmp/garland_mechanism_diag.py
```

The corrected two-pass diagnostic sums each wearable's empirical
per-step, per-anomaly-type rate over the wearable-steps composing each
emission group, including zero-count groups. The full-run per-agent summary
was mean **2.443** tokens, SD **5.933**, variance-to-mean **14.410**,
top-decile share **73.434%**, zero fraction **84.9%**, and maximum **25**.
The heterogeneity-adjusted emission dispersion was **6.320** across 15,288
groups, versus **5.789** for the population-rate statistic. Excluding the
first simulated day, those values became **1.079** and **1.029**,
respectively. The corrected diagnostic therefore does not support publishing
agent heterogeneity as the sole explanation; the small full-run difference is
not treated as a mechanism claim.

The startup comparison was:
The `all steps` and configured-warm-up rows are full-run views of this
`burned_in` run; the post-day row is the settled view.

| sample | population VMR | emission dispersion | window dispersion |
| --- | ---: | ---: | ---: |
| all steps | 47.597 | 5.789 | 9.521 |
| excluding first 288 steps (one day) | 1.201 | 1.029 | 1.214 |
| excluding configured warm-up (0 steps) | 47.597 | 5.789 | 9.521 |

The first day is therefore the dominant contributor to the committed
full-run over-dispersion. The post-startup values, rather than the
full-run values, are the operating point for the settled null process.

For the settled process, the same run measured mean lag-1 anomaly-indicator
autocorrelation **0.277**, mean consecutive anomalous run length **1.356**
steps, and mean geometric independence expectation **1.008** steps using
each agent's own rate. The population background series had mean **1.212**
tokens per step and full-run VMR **47.597**. The post-startup population VMR
was **1.201**. Thus the data support a startup transient and residual
within-agent persistence as contributors; they do not support the shared
activity sinusoid as the dominant linear common-mode explanation.

The diurnal profile also shows that the unstratified full-run population VMR
is not a settled diurnal operating point. Full-run within-hour VMR was
**127.843** at hour 0 and ranged from **0.786** to **4.197** for the other
hours. Correlation of population background count with activity level was
**−0.034**, while correlation with the step-to-step activity change was
**0.032**. The settled within-hour values were generally near one to four,
so diurnal stratification removes much of the remaining variation after the
startup day, but not all of it.

Two one-change ablations used the same in-session script:
These are full-run measurements from a `burned_in` run; settled status must be
read from the marker fields for the corresponding run.

| run | emission dispersion | window dispersion | population VMR |
| --- | ---: | ---: | ---: |
| unmodified | 5.789 | 9.521 | 47.597 |
| activity held constant | 6.764 | 10.998 | 49.497 |
| circadian amplitudes zeroed | 5.635 | 9.287 | 44.648 |

These ablations do not identify either activity level or circadian amplitude
as the dominant driver of the settled over-dispersion. Remaining temporal,
spatial, and detector-transient causes are open questions.

Plume exposure uses the existing concentration gate of `> 0.01`. Exposed
plume observations are classified as respiratory before the generic
multi-system fallback when they are fever-free; late-stage infection remains
febrile or multi-system because it includes a temperature increase.
