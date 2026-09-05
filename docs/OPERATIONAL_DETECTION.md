# Operational detection measurements

## K-anonymity dilation bases

Spatial dilation can use one of three configured population bases:

- `residents` uses the existing resident population in each cell. It is
  retained as a reproducibility and negative-control setting.
- `observed_devices` is the operational default. The aggregator records all
  protocol-visible token arrivals, including dummy packets, in a trailing
  window set by `time_window_steps` (12 five-minute steps by default in the
  base configuration). It divides the observed traffic
  by the configured dummy rate and window length, then subtracts a configurable
  Poisson-style margin (`margin_factor * sqrt(observed)`) before converting the
  result to a conservative device estimate. Under-estimation causes additional
  dilation; over-estimation would weaken the intended anonymity property.
- `true_devices` is an evaluation-only oracle reference for measuring estimator
  over-dilation. It is never selected by default and must not be used as
  operational protocol truth.

The default margin factor is `0.5`, a deliberately smaller conservative margin
than the earlier exploratory value. The estimator also normalizes early-run
traffic by the history actually available rather than the full nominal window.
It is a historical occupancy estimate, not an instantaneous count: devices
moving between cells can make venue clustering lag under schedule mobility.
The observed-traffic estimator defaults to `time_window_steps`, the response
window, rather than a full day. A shorter window buys currency about current
occupancy at the cost of a noisier estimate. Genuine anomaly-token arrivals are subtracted before
inverting the dummy rate. The estimator has a storage and computation cost
proportional to the number of active cells that have emitted traffic during its
trailing window. Dummy traffic is already protocol-visible, so this estimate does not
inspect agent objects, wearable adoption attributes, or other model-side
truth. Wider respondent-based zones can increase the number of reachable
devices and therefore the epsilon spent on each broadcast; that trade-off is
intentional.

If the conservative estimate remains below `k_min` after the backend's bounded
maximum dilation, the trigger is suppressed instead of being broadcast over the
entire city. Suppressed triggers consume no response epsilon and cannot produce
detection events or disambiguation asks. Metrics report issued broadcasts and
suppressed-for-insufficient-anonymity triggers separately, including the
estimate reached at suppression.

After a broadcast, the aggregator measures whether at least `k_min` devices
responded. The strict respondent-reading enforcement switch is
`enforce_release_k_anonymity`, and it is **off by default**. The under-k
condition, its positive-reply k coverage, and its epsilon burn are still
reported exactly as measured regardless of that switch. With randomized
response, the released positive count stayed under 1σ from the null across
the measured `p` values, so enforcing k on positive replies gates on coin
noise rather than reachable-device anonymity; it suppressed every release in
the mill measurement. If enforcement is enabled, under-k releases are
suppressed after responses have already transmitted and their response
epsilon is not refunded. Metrics report the under-k release count and the
epsilon burned on those releases separately.

Respondent-based dilation has a separate feasibility cost. If the reachable
population's wearable adoption cannot reach `k_min` within the spatial bound,
most or all triggers are suppressed before a broadcast is issued, so the
scenario can detect nothing. This was measured as 100% release suppression in
the mill archetype before the release gate was made default-off, and appears
as total non-detection in small scenarios. The default-off release gate does
not remove this pre-broadcast infeasibility suppression; it only permits
classification after a feasible response round has returned under `k_min`.

The same distinction is visible in runtime measurements on the calibrated
scaling scenarios. On an otherwise idle machine, the 1,000-agent, three-step
benchmark averaged 95 ms per step with resident-based dilation and 189 ms
with observed-device dilation; the corresponding 5,000-agent, ten-step run
averaged 565 ms and 2,129 ms per step before test instrumentation. Under the
benchmark helper's memory tracing, the 1,000-agent observed-device run
averaged 4,362 ms per step. The main worktree measured 95 ms and 522 ms for
the direct scenarios, respectively. The quick benchmark retains its 2,000 ms
resident-basis budget. Cross-basis regression guards run the resident and
observed-device cases in the same process and require the respondent runtime
to remain within 8x of the resident runtime, with a generous absolute
catastrophe ceiling. The local three-step benchmark ratio was about 3x
(1,019 ms versus 333 ms), while the ring-search regression was about 10x.
These relative guards expose that regression without treating one machine's
wall-clock speed as a portability contract; the absolute ceilings are only
catastrophe guards. Repeated ten-step local benchmark runs measured a 3.0–3.5x
ratio, which is why the guard uses more steps than the earlier three-step
smoke measurement.

## Second-round disambiguation

The optional disambiguation layer is an interpretation aid, not validation.
The aggregator asks only from protocol-visible shape. A narrow, persistent,
weakly confirmed cluster can raise `RECENT_ADOPTION`; broad simultaneous
activity can raise `AMBIENT_HEAT`. The former implementation gated the ask on
model-side device age, which was oracle validation and made previously
published disambiguation numbers optimistic. The
`min_onboarding_wearables_in_zone` field was removed as a breaking change to a
default-off experimental feature. No device reports age, adoption step, or
other per-device metadata to the predicate.

Acknowledgements are automatic and content-free: they indicate only that a
device is reachable in the queried zone. They are released as a noised,
zone-level count subject to the existing `k_min` floor. An ack is separate from
the human answer. A reachable person may approve yes or no, or provide no
answer. Non-response is free, never inferred as a negative, and expires as an
unresolved hypothesis. Both approved answer arms are charged separately from
the round-one response budget. Reported yes/no counts are
randomized-response perturbed rather than raw human answers; an affirmation
count is contextual evidence, not ground truth or validation. Each ask is
scored separately against model-side benign ground truth as well-founded when
the cause matches, unfounded when a benign instance is present but the cause
does not match, or unscored when no benign ground truth is available for that
zone. The counts obey `well_founded + unfounded + unscored == queries issued`.
Unfounded asks and their answer-plus-ack epsilon are reported separately from
unscored asks and their epsilon. Neither bucket is deliberately penalized in
`discrimination_score` or existing hazard metrics. Before revisiting that
choice, evaluate unfounded-ask rates under realistic confounder mixes, epsilon
burned on unfounded asks, and whether an unfounded ask should eventually carry
a cost. These mechanics are simulation
measurements, not a formal DP proof or a claim of real encryption.

## Privacy accounting and proofs owed

The default content round uses a truthful device reply plus one noisy aggregate
count per broadcast. This is a deliberate trust-model change: the aggregator
sees truthful device replies, so a device has no deniability against that
aggregator. The content round is therefore central/trusted-curator rather than
a local mechanism; protection applies to the released count, not to an
individual reply against the aggregator. Randomized response remains available
as an explicit compatibility mechanism for historical runs and comparisons.

The aggregate count has sensitivity one and uses Laplace scale
`1 / aggregate_count_epsilon`. The clamp bound is the protocol-visible
respondent-population estimate produced during dilation, not a model-side
wearable count. If that estimate undercounts the true matching devices, the
release saturates at the estimated population; the unbounded true count is
retained only as evaluation metadata, so the saturation is measurable rather
than silently substituted into the protocol. Rounding and clamping the release
to the achievable range are post-processing, but clamping folds negative noise
onto zero and therefore biases releases upward near zero. Detection requires the
released count to exceed a one-sided Laplace evidence threshold. For scale
`b`, the null upper tail is `0.5 exp(-t / b)`; the threshold is the ceiling of
`b * log(1 / (2 * false_release_rate))`. The minimum releasable cluster size is
therefore threshold + 1. With the default false-release rate of 0.05:

| Aggregate epsilon per release | Noise scale | Evidence threshold | Minimum releasable count |
|---:|---:|---:|---:|
| 0.2 | 5.0 | 12 | 13 |
| 0.5 | 2.0 | 5 | 6 |
| 1.0 | 1.0 | 3 | 4 |
| 2.0 | 0.5 | 2 | 3 |

Lower per-release epsilon buys a noisier, more private released count but
raises this floor. A release of zero is reported as `no_cluster`; a positive
release at or below the threshold is reported as `cluster_below_floor`.
Neither is detection evidence.

In the fixed staged plume geometry, the 100-agent toxin guard required raising
the plume release rate from 5 to 100 to produce a toxin detection above the
default floor. The baseline rate already produced occasional true matching
clusters as large as 10, but did not produce a toxin detection in the
aggregate classification path. The smallest tested stronger-plume
configuration produced released toxin evidence with a count of at least 4.
The paired default-mode guards therefore cover both sides of the floor: the
baseline toxin case remains suppressed, while this stronger but
scenario-faithful plume produces a toxin detection.

The minimum toxin truth/exposure gate is calibrated from physiology rather than
an arbitrary concentration. The default `minimum_respiratory_delta_bpm` is
2.0 bpm, which inverts the plume response
`12 * c / (c + 0.5)` to a concentration gate of `c > 0.1`. The same
evaluation-only gate drives exposure labels, zone-local toxin truth, exposure
metrics, and scoring; it never enters tokens, broadcasts, aggregation,
dilation, or question content. The physiology contribution uses a separate
negligibility floor derived by the same inverse: `0.1` bpm maps to
approximately `c > 0.0042`. This keeps the perturbation model continuous while
avoiding sensor movement and `TOXIN` cause provenance for sub-perceptual doses.
`toxin_exposure_gate_mode: legacy_0_01` explicitly reproduces pre-calibration
results and uses `c > 0.01` for both evaluation and physiology floors; it is
not recommended for new scenarios.

At the default operating point, a toxin cluster must reach about four devices
in a zone before the content round can report toxin evidence. Smaller
per-release epsilon values buy more noise/privacy but raise this floor. The
maintained `scripts/calibrate_toxin_footprints.py` harness reports the physical
footprint and implied wearable count for candidate release rates.

Aggregate mode charges its configured epsilon once per released count, not once
per replying device. RR mode retains per-device response composition. Both
the released count and the true matching count are reported; the latter is
evaluation-only and never drives detection or disambiguation. The estimated
respondent population, rather than individual device truth, supplies the
protocol-visible denominator for aggregate disambiguation fractions and the
release k-anonymity check. Thus aggregate-mode k measures the estimated
population over which the count is released, while RR-mode response metrics
continue to describe individual replies.

The summary reports the selected content mechanism and keeps
randomized-response deniability quantities when RR is selected: the
probability that an unaffected device reports positive,
`0.5 * (1 - randomized_response_p)`, and the selected per-response epsilon.
Under aggregate mode, per-device response epsilon is explicitly zero. The
summary reports both the configured aggregate epsilon per release and the
composed aggregate release total. Composition uses the tighter basic or
advanced expression; for one release the charge is exactly the configured
epsilon. This remains indicative accounting, not a proof, because broadcasts
are data-triggered.

The floor changes the interpretation of small-cluster measurements. In the
100-agent, 1,200-step toxin-only staged CI scenario (`threshold_m=2`,
residents basis), aggregate mode produced 243 releases: the median true
matching cluster was 1, 102 releases had zero genuinely anomalous devices,
and only 26 had four or more. No toxin detection was therefore expected from
the aggregate content round. The same code and scenario under historical RR
produced two toxin true-positive events, affecting 1 and 4 agents, with
`time_to_detection_toxin_steps=238`; that first detection rested on a
single-device confirmation. Total epsilon was 243 for aggregate releases
versus 832.8 for RR responses. This is a measured signal-loss tradeoff, not a
reason to lower the evidence floor.
The mechanism-derived basis uses `ln((1+p)/(1-p))`; the legacy basis retains
the historical configured constant for reproduction. The planar channel reports
`1 / laplace_scale` as a geo-indistinguishability parameter per metre
(metres⁻¹), separately from response epsilon. It is not added to the response
total because the channels use different metric spaces and
indistinguishability notions, and this testbed does not justify a composed
bound.
JSON summaries preserve non-finite accounting values with explicit marker
objects such as `{"__garland_nonfinite__": "Infinity"}`; this is strict
JSON and distinguishes an unbounded value from `null` or an absent field.

### Historical randomized-response tradeoff

The settled 1,152-step sweep covered mill and college archetypes under benign
and seeded arms. Across both archetypes and both arms, released positive-reply
counts stayed under 1σ from the randomized-response null at every tested
truthfulness value: median signal excess was 0.07–0.86σ and p90 was
1.5–2.4σ. Seeding an outbreak did not materially change that result.

| `p` | ε per response | Unaffected positive probability | Mill release suppression (benign / seeded) | College release suppression (benign / seeded) |
|---:|---:|---:|---:|---:|
| 0.75 | 1.946 | 0.125 | 1.00 / 1.00 | 0.90 / 0.89 |
| 0.50 | 1.099 | 0.250 | 0.79 / 0.77 | 0.47 / 0.49 |
| 0.25 | 0.511 | 0.375 | 0.38 / 0.39 | 0.23 / 0.24 |
| 0.10 | 0.201 | 0.450 | 0.32 / 0.22 | 0.14 / 0.13 |

For the historical RR mechanism, lowering `p` improves release feasibility and reduces
the released count's excess over the null. The default is therefore `p=0.5`,
rather than a lower value: below 0.5, cheaper responses do not rescue the
content round's signal. For a 150–220-device dilated zone, per-device
randomized response spends epsilon for well under one sigma of aggregate
signal on its own. The content round therefore cannot carry detection evidence
by itself; the token threshold carries that role.

The sweep did not capture detection true positives or latency: its harness
requested summary keys that do not exist. No claim is made about detection
latency or TP rate versus `p`.

Proofs owed before making formal privacy or security claims:

- Randomized response is a per-response local-DP mechanism only. Nothing here
  proves privacy for repeated queries about the same person's correlated
  physiology.
- The advanced-composition calculation assumes a query sequence independent of
  the data. Broadcasts here are triggered by the data, so reported totals are
  indicative accounting, not a proven bound.
- K-dilation counts a population; it is not an anonymity proof. The respondent
  gap is measured and reported, not bounded.
- Tokens are plaintext tuples in this simulation; there is no encryption.
- The geo channel is reported separately and is not included in the composed
  response budget.
- The aggregate count is a sensitivity-one Laplace mechanism in isolation, but
  broadcasts are triggered by the data. Composition across adaptive releases
  therefore remains unproved; reported totals are indicative accounting, not a
  formal bound for the full protocol.
- Truthful content replies are visible to the central aggregator. No claim is
  made that the aggregate mechanism protects an individual device from that
  aggregator.

### Disambiguation ask-quality evaluation

The operator-run `scripts/disambiguation_ask_eval.py` measured the authored
`examples/disambiguation_evaluation.yaml` scenario with seed 42,
`PYTHONHASHSEED=0`, 2,000 agents, 1,152 steps, 288 world-settling steps, and
both hypotheses enabled. It is deliberately not wired into pytest or CI.

#### Before: single-step absolute breadth, no ask budget

- **mix+onboarding**: `broadcasts=1980`, `asks=1133`,
  `asks_per_broadcast=0.57`; `well_founded=82 (0.072)`,
  `unfounded=614 (0.542)`, `unscored=437 (0.386)`;
  `recent_adoption: asks=61 wf=29 uf=7 us=25`;
  `ambient_heat: asks=1072 wf=53 uf=607 us=412`;
  disambiguation epsilon `198.4 of 387.3 = 51.2%`;
  `unfounded_ask_epsilon=115.8`, `unscored_ask_epsilon=67.0`.
- **mix only**: `broadcasts=1778`, `asks=902`,
  `asks_per_broadcast=0.51`; `well_founded=42 (0.047)`,
  `unfounded=40 (0.044)`, `unscored=820 (0.909)`;
  `recent_adoption: asks=35 wf=0 uf=7 us=28`;
  `ambient_heat: asks=867 wf=42 uf=33 us=792`;
  disambiguation epsilon `165.7 of 349.1 = 47.5%`;
  `unfounded_ask_epsilon=5.9`, `unscored_ask_epsilon=149.9`.
- **mix+outbreak**: `broadcasts=2039`, `asks=1113`,
  `asks_per_broadcast=0.55`; `well_founded=68 (0.061)`,
  `unfounded=614 (0.552)`, `unscored=431 (0.387)`;
  disambiguation epsilon `192.5 of 390.4 = 49.3%`.
- **no ground truth**: `broadcasts=1344`, `asks=939`,
  `asks_per_broadcast=0.70`; `well_founded=0`, `unfounded=0`,
  `unscored=939 (1.000)`; disambiguation epsilon
  `151.7 of 267.3 = 56.8%`; `unfounded_ask_epsilon=0.0`,
  `unscored_ask_epsilon=151.7`.

The follow-up fires on roughly half of broadcasts and consumes roughly half
of total epsilon. `recent_adoption` precision is `29/36 = 80.6%` over
scorable asks in mix+onboarding; `ambient_heat` precision is `53/660 = 8.0%`
over scorable asks and accounts for 95% of all asks. Precision means
`well_founded / (well_founded + unfounded)`; unscored asks are excluded, not
counted against it. Removing onboarding drops `recent_adoption` precision to
`0/7` scorable. A seeded outbreak barely changes the ask rate or split, so
benign-explanation questions are currently asked at the same rate during a
real hazard. The no-ground-truth control is 100% unscored, whereas a two-way
split would have published it as 100% unfounded.

The decision remains reporting-only: unfounded asks stay out of
`discrimination_score`, because averaging a penalty over both hypotheses would
hide that one predicate is informative and the other spends half the privacy
budget at 8% precision.

#### Why breadth alone failed: the baseline was measured on an unsettled world

Instrumenting the per-bin breadth series explains those numbers. Breadth is
6-23 distinct trigger footprints per bin during the world-settling day, and
1-3 per bin once the world is settled (mean 1.48, maximum 6). The absolute
`min_breadth: 4` floor was therefore calibrated against the un-settled startup
period: 13 of the 16 bins that ever cleared it fall inside world settling, and
because a passing bin turned every broadcast in that bin into an ask, a handful
of cold-start bins produced 1,072 asks. `ambient_heat` was not measuring an
ambient cause at all; it was measuring the fleet turning on.

A relative test alone does not fix this. An exponentially weighted baseline
that learns during settling is seeded near 12 and decays too slowly to be
exceeded afterwards, which silences the hypothesis for the rest of the run,
including the heat wave. The baseline is a statement about what a run normally
produces, so it must not learn from a period that is not a valid operating
point: breadth bins inside `world_settling_steps` update neither the baseline
nor the sustained-window history. Asks themselves are not suppressed during
settling.

#### After: sustained relative breadth and an explicit ask budget

`AMBIENT_HEAT` now requires the last `min_breadth_windows` recorded broadcast
bins to each clear the absolute floor and exceed `breadth_ratio` times the
channel baseline as it stood before that bin. The scenario sets
`min_breadth: 3` (calibrated to settled-world breadth, not cold-start
breadth), `min_breadth_windows: 2`, `breadth_ratio: 2.0`,
`breadth_baseline_alpha: 0.05`, and `ask_epsilon_budget: 40.0`. The budget is
checked against epsilon already spent immediately before each ask, so the
channel may overshoot by at most one ask's cost; `disambiguation_max_ask_epsilon_delta`
publishes that worst-case single-ask cost. `RECENT_ADOPTION` is unchanged.

- **mix+onboarding**: `broadcasts=1980`, `asks=123`,
  `asks_per_broadcast=0.062`, `suppressed_by_budget=37`;
  `well_founded=86`, `unfounded=4`, `unscored=33`; `precision=0.956`;
  `recent_adoption: asks=57 wf=29 uf=4 us=24 precision=0.879`;
  `ambient_heat: asks=66 wf=57 uf=0 us=9 precision=1.000`;
  disambiguation epsilon `40.34 of 229.24 = 17.6%`;
  `unfounded_ask_epsilon=0.74`, `unscored_ask_epsilon=8.85`;
  `max_ask_epsilon_delta=1.43`.
- **mix only**: `broadcasts=1778`, `asks=69`,
  `asks_per_broadcast=0.039`, `suppressed_by_budget=0`;
  `well_founded=34`, `unfounded=7`, `unscored=28`; `precision=0.829`;
  `recent_adoption: asks=35 wf=0 uf=7 us=28 precision=0.000`;
  `ambient_heat: asks=34 wf=34 uf=0 us=0 precision=1.000`;
  disambiguation epsilon `29.09 of 212.47 = 13.7%`.
- **mix+outbreak**: `broadcasts=2039`, `asks=116`,
  `asks_per_broadcast=0.057`, `suppressed_by_budget=26`;
  `well_founded=89`, `unfounded=4`, `unscored=23`; `precision=0.957`;
  `recent_adoption: asks=56 wf=29 uf=4 us=23`;
  `ambient_heat: asks=60 wf=60 uf=0 us=0`;
  disambiguation epsilon `40.83 of 238.77 = 17.1%`.
- **no ground truth**: `broadcasts=1344`, `asks=66`,
  `asks_per_broadcast=0.049`; `unscored=66 (1.000)`; `precision=None`;
  `recent_adoption: asks=54`, `ambient_heat: asks=12`;
  disambiguation epsilon `22.71 of 138.29 = 16.4%`.
- **mix+onboarding, tight budget** (`ask_epsilon_budget: 5.0`):
  `asks=15`, `asks_per_broadcast=0.008`, `suppressed_by_budget=145`;
  `well_founded=7`, `unfounded=0`, `unscored=8`; `precision=1.000`;
  disambiguation epsilon `5.14 of 194.03 = 2.6%`;
  `max_ask_epsilon_delta=1.04`.

Asks fall from 0.57 to 0.062 per broadcast and the channel's share of total
epsilon from 51.2% to 17.6%. `ambient_heat` drops from 1,072 asks at 8.0%
precision to 66 asks with no unfounded ask in any variant, and it still fires
without benign ground truth (12 unscored asks in the control), so the gate is
not oracle-dependent. `recent_adoption` is materially unchanged: 0.879
precision with onboarding and 0.000 over 7 scorable asks without it, the same
honest failure as before.

The budget binds where it is meant to: the channel spends 40.34 against a 40.0
budget, an overshoot of 0.34 within the documented one-ask allowance of 1.43,
and at a 5.0 budget it spends 5.14 against a 1.04 allowance while suppressing
145 asks. Suppressed asks are never issued, never answered, and never scored,
so `well_founded + unfounded + unscored == queries issued` still holds exactly.

Two cautions on the precision figures. `ambient_heat` reaching 1.000 is
measured in a scenario whose only broad benign source is a day-long heat wave,
so it should be read as "the surviving asks land inside the heat wave" rather
than as a precision claim that transfers to other worlds; and `min_breadth: 3`
is calibrated to this scenario's settled breadth, so it is a scenario
parameter, not a recommended default. The library default remains 4.

The reporting-only decision is unchanged, and the case for revisiting it is
now weaker rather than stronger: unfounded asks are 4 of 123, and the epsilon
they burn is 0.74. What still needs evaluation is whether a genuine hazard
ought to move the ask rate at all - a seeded outbreak still produces almost
the same asks as the benign mix.

## Benign confounder engine

The implemented exposure layer, warrant classes, and privacy boundary are
specified in [`EVENT_CATALOGUE.md`](EVENT_CATALOGUE.md). Warrant reporting is
additive to the historical hazard metrics, and ask scoring accepts any
overlapping matching benign instance rather than only the dominant instance.

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
are model-side metrics; they are not added to plaintext token tuples or interpreted
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
are model-side metrics; they are not added to plaintext token tuples or interpreted
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
count only model-side provenance recorded at token emission: toxin status uses
the configured dose-derived concentration gate (default `> 0.1`), and disease
status uses the agent's SEIR state at emission. The provenance is not part of
`EncryptedToken`, is not available to the aggregator, and cannot affect
detection, privacy responses, or query behavior. The fragmentation breakdown
shows whether affected tokens split across anomaly types or fail to form a
same-zone/type group large enough to reach `threshold_m`. Toxin detection
events also report the evaluation-only number of dosed agents among the
affected count, plus a summary count of true positives with fewer than two
dosed agents; these do not change protocol-visible detection semantics.

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

## Calibrated cold-start covariance prior

Each `BaselineTracker` starts with a diagonal covariance prior. GARLAND's core
channel values are calibrated from the model's own benign physiology: 100
devices were matured for five benign days, and residuals were measured on day
six using the live activity level and activity jitter. These values represent
**mature-tracker benign residual variance**, not raw observation variance.
They are a simulation-testbed calibration and are not a claim about variance
in real wearable devices.

The calibration corrects a cold-start fever-blindness defect in the former
shared prior. A flat variance of 10 made a 0.8 °C body-temperature excursion
only about 0.25 prior standard deviations, while the calibrated
body-temperature prior makes the same excursion more than five standard
deviations. The reproducibility harness is:

```bash
PYTHONPATH=src python scripts/coldstart_variance_check.py
```

The harness prints measured benign residual variance against each committed
prior so physiology-model drift is visible. This calibration does not alter
the covariance-prior-strength mechanism and makes no formal differential
privacy, encryption, anonymity, or security claim.

## Detection power by observation width

The episode metrics above measure the *system*: whether a zone alarmed, and how
long it took. In a mixed-modality fleet they cannot answer whether adopting a
sensor subsystem bought anything, because they are not keyed to what any
individual was wearing. `summary()["detection_power"]` is, and is measured at the
sensing layer — per agent-epoch, before K-anonymity dilution and aggregation.

An epoch's **effective width** is the number of channels that were both present
and unmasked when it was scored. Structural missingness (a subsystem nobody
adopted) and duty-cycle masking (a subsystem that yielded nothing this epoch, or
whose battery is flat) both reduce it, so one person moves between width buckets
over a day. Epochs are counted only when the detector could have alarmed on them,
so an agent still in baseline warm-up is neither a scored epoch nor a silent one.

```bash
garland --config examples/detection_power_town.yaml --no-plots \
  --output-dir output/detection_power_town
garland sweep --sweep-config examples/detection_power_adoption_sweep.yaml
garland sweep --sweep-config examples/detection_power_ladder_sweep.yaml
garland sweep --sweep-config examples/detection_power_density_sweep.yaml
```

| Block | What it answers |
|-------|-----------------|
| `width_buckets` (1–5, 6–12, 13–24, 25+) | Does a wider vector detect more, sooner? `true_positive_rate` and `mean_detection_latency_steps` per bucket. |
| `width_buckets[*].false_positive_rate` | Is the null rate flat in width? A rate that climbs with width falsifies the degrees-of-freedom threshold calibration rather than showing a detection gain. |
| `devices[*]` | How much of each subsystem's channel budget survives duty cycling (`observed_channel_fraction`, `reporting_epoch_fraction`), and its owners' outcome rates. |
| `channel_ablation` | Is detection collective? Per-channel `marginal_contribution` over the alarms that channel was present for. |

Two cautions on reading it. Width buckets are not randomized arms: the people
wearing more sensors are self-selected by the adoption model, so a bucket
difference is an association within one run and the adoption sweep is the
controlled comparison. And latency here is per person — from the epoch an agent
first became hazard-affected to its first emitted token — which is a lower bound
on the operational latency the aggregation layer reports.

The ablation is off by default (`detection_power.channel_ablation_rate: 0.0`); it
costs one extra Mahalanobis evaluation per observed channel on each sampled
alarming epoch. It probes the instant detector only, because a single-epoch
re-score cannot say what a path-dependent CUSUM would have done. Because a
channel is credited only for alarms it was present for, a rarely-observed channel
can show a large contribution on a handful of evaluations; read
`alarms_evaluated` before believing a contribution.

### Host-stratified detection

When `hosts.enabled` is true, `summary()["detection_power"]["host_groups"]`
reports sensing-layer outcomes for diabetic, frail-elderly, law-enforcement,
assistive-need, and complementary `general` groups. Each group's disease and
toxin true-positive rate is the fraction of infected or exposed agent-epochs
that emitted a token; its false-positive rate is the fraction of clean,
reporting agent-epochs that emitted one. Detection latency is measured from the
model-side oracle onset of that hazard kind to the first token for that agent.
These are oracle-truth measurements, not information available to the privacy
protocol, and they are measured before aggregation, K-anonymity dilution, or
broadcasting.

Host groups overlap by design: a diabetic frail-elderly person contributes to
both groups. `general` excludes every flagged phenotype, so counts across
groups must not be summed. Epochs with effective width zero are excluded from
the scored, hazard, and clean denominators.

## Holding the quiet-epoch alarm rate flat in width

The degrees-of-freedom conversion in `garland.thresholds` keeps the alarm rate
fixed across widths only if the scored residuals are jointly Gaussian. Much of
the wide fleet is not: step count, cough rate, wheeze and crackle burden,
bladder impedance and ectopy burden are floored at zero and right-skewed, so the
null Mahalanobis statistic has a heavier right tail than chi-square. Measured on
a hazard-free, confounder-free 600-agent run with every subsystem adopted, the
cut that should flag 1.56% of quiet epochs flagged 4.5% at 6–12 channels, 5.9%
at 17–20 and 9.4% at 21–30 — a wider vector looked more sensitive largely
because it alarmed more often on nothing.

`alarm_calibration` corrects that empirically. Over a window
(`start_step`–`end_step`, default steps 144–720) the fleet accumulates the ratio
of each scored distance to the chi-square cut for its width, in a histogram per
width bucket, then reads off the upper quantile matching the configured target
rate. That quantile becomes a multiplicative scale on the cut and is frozen for
the rest of the run. On the same null run the post-freeze rates were 0.0102,
0.0121, 0.0136 and 0.0175 across the 6–12, 13–16, 17–20 and 21–30 widths against
a 0.0156 target — flat in width rather than climbing fivefold.

Three properties matter for reading it:

- The scale is floored at 1.0 and capped at `max_scale`, so calibration only ever
  makes agents less trigger-happy, and a narrow fleet already at target is left
  essentially untouched (learned scale 1.03 on the 2K town scenario).
- It is fleet-level and frozen, not per-agent and adaptive. A per-agent
  Robbins-Monro variant of the same correction halved the false-positive rate on
  the town scenario and also lost the outbreak; a cut calibrated once against a
  mostly quiet reference population cannot desensitize itself during an episode.
- The window must sit in quiet time. Hazard-affected epochs inside it inflate the
  learned scale and cost sensitivity later; the calibrator has no hazard oracle
  and cannot detect this for you.

The correction costs real sensitivity, and the honest comparison is at matched
null rate. Scoring the same town epochs both ways (the identical epochs, so no
run-to-run divergence), the 6–12 bucket moved from FPR 0.0265 / TPR 0.083 to FPR
0.0066 / TPR 0.042: a quarter of the false alarms for half the true ones, and
wide agents remain about five times as sensitive as 1–5 agents at the same false
alarm rate. Zone-level `warranted_detections` fell as well (7 → 1 on seed 42),
because `privacy.threshold_m` was chosen against the inflated token volume; the
aggregation thresholds are the next thing to recalibrate, not evidence that the
per-epoch correction is wrong.

`detection_power.alarm_calibration` reports `frozen`, `calibration_epochs` and
the per-bucket `scales`. Note that `width_buckets[*].false_positive_rate` pools
the whole run, including the uncalibrated window before the freeze, so it
understates the correction; disable it with `--no-alarm-calibration` for the
uncorrected baseline.

## Recalibrating the aggregation layer against the corrected token rate

`privacy.threshold_m` was chosen against the pre-calibration token volume, so it
had to be re-picked. Measuring where zone triggers actually fall on the 2K town
scenario, phase by phase, produced three results that matter more than the
constant itself.

**Most triggers were startup, not signal.** With the trigger count at 5, 2,365
of the run's 3,139 zone triggers landed in the 300 settling steps and a further
661 before the alarm scales froze at step 720; the post-freeze quiet rate was
0.24 triggers/step against 0.11 during the plume. Those early triggers come from
cuts the run is in the middle of establishing are miscalibrated, and each one
spends response epsilon. `alarm_calibration.defer_broadcasts_until_frozen`
(`--defer-broadcasts-until-calibrated`) therefore withholds broadcasts until the
scales freeze, while continuing to ingest tokens so a zone can still trigger on
its in-window history once the gate lifts. On the town at
`wearable_fraction` 0.6 that took broadcasts from 9,816 to 899 while keeping 20
of 27 warranted detections — precision 0.27% → 2.2%, and an order of magnitude
less epsilon spent, all of it after the fleet knows its own cut.

It is off by default and set per scenario, because it is only sound when the run
reaches the freeze and the hazards arrive after it: a run shorter than `end_step`
never broadcasts at all under the gate, and a hazard that starts and ends inside
the calibration window is missed entirely. `examples/detection_power_town.yaml`
enables it because its plume starts at step 864 and its outbreak at 1,152, both
after the step-720 freeze; `examples/staged_onset.yaml` deliberately does not,
because its disease arm is detected before then.

**The trigger count trades volume against detections smoothly, and 8 is the
knee.** Gated, at `wearable_fraction` 0.6, warranted detections / broadcasts ran
26/2562 at a count of 3, 20/899 at 5, 17/382 at 8 and 4/201 at 12. Above 8 the
plume survives but the outbreak stops being detected at all; `threshold_m: 8` in
`examples/detection_power_town.yaml` keeps 85% of the detections a count of 5
finds for 42% of its broadcasts. `time_window_steps`, the dilation margin and the
aggregate-count evidence floor were left alone: the suppression rate moved only
between 0.069 and 0.112 across the whole sweep, so K-anonymity dilution is not
what is binding here.

**What binds is device density, not the threshold.** At the town's authored
`wearable_fraction` of 0.15 the layer finds 1 warranted detection at any trigger
count from 3 to 20, because a zone holds too few devices for a hazard to clear
any count that background alarms do not also clear. At 0.6 the same scenario and
seed finds 17–26. Read zone-level detection numbers from this scenario as a
statement about adoption density first and the trigger count second.

`zone_threshold_calibration` (off by default, `--zone-threshold-calibration`)
re-derives the count from the quiet-window token counts the aggregator itself
sees, targeting `false_trigger_rate`, and freezes it. It is the mechanism to
prefer over hand-picking a constant when the fleet density is unknown, with two
limitations: it learns one pooled fleet-level count, so a dense zone and a sparse
one get the same one, and a hazard inside the calibration window inflates what it
learns exactly as it does for the alarm scales.

## The population ladder: where the zone layer starts working

With the aggregation layer recalibrated, the ladder
(`examples/detection_power_ladder_sweep.yaml`, seed 42, gate on,
`threshold_m: 8`, `wearable_fraction` 0.15) climbs 2K → 10K → 25K residents on
the *same* 2 km grid, so each rung is both a larger population and a denser one.

| Residents | Broadcasts | Warranted | Toxin TP | Disease TP | Toxin TTD | Disease TTD | Discrimination |
|-----------|-----------|-----------|----------|------------|-----------|-------------|----------------|
| 2,000 | 21 | 2 | 2 | 0 | 131 | — | undefined |
| 10,000 | 581 | 55 | 47 | 8 | 81 | 97 | 1.00 |
| 25,000 | 1,949 | 221 | 181 | 40 | 75 | 84 | 0.997 |

Three things to read from it:

- **The outbreak becomes detectable between 2K and 10K.** The 2K rung finds the
  plume twice and the outbreak never — a plume raises every device in a cell at
  once, an outbreak raises a few devices scattered across cells, so the outbreak
  needs more devices per zone to clear the same trigger count. Nothing about the
  detector changes across rungs; `mean_effective_width` is 4.79 at all three.
- **Cost per detection does not degrade with scale.** Broadcasts per warranted
  detection run 10.5 / 10.6 / 8.8, and response epsilon is one unit per
  broadcast, so the privacy bill grows with detections rather than with
  population. Latency improves (toxin 131 → 75 steps).
- **Attribution stays clean while the false-positive share does not.** The
  discrimination score is ~1.0 at both detecting rungs, but
  `unexplained_detection_rate` is 0.26 at 10K and 25K against 0.0 at 2K: the
  rungs that detect anything also emit background detections with no assignable
  cause. That is the number to watch when the ladder goes to 250K.

### Which of population and adoption the gain belongs to

Each rung moves resident population and wearers per zone together, so
`examples/detection_power_density_sweep.yaml` holds the population at 2,000 and
sweeps `wearable_fraction` instead (same seed, gate on, `threshold_m: 8`):

| Wearable fraction | Wearers | Broadcasts | Warranted | Disease TP | Toxin TTD |
|-------------------|---------|-----------|-----------|------------|-----------|
| 0.15 | 300 | 21 | 2 | 0 | 131 |
| 0.30 | 600 | 179 | 8 | 1 | 93 |
| 0.60 | 1,200 | 435 | 45 | 1 | 77 |

Read the two tables together and, **for broadcasts and plume detections**, it is
the absolute number of wearers that the zone layer responds to — not the resident
population and not the adoption fraction. Runs holding wearers fixed while
population and adoption share both move land together:

| Wearers | Config A | Config B | Broadcasts A / B | Warranted A / B |
|---------|----------|----------|------------------|-----------------|
| 600 | 2,000 @ 0.30 | 4,000 @ 0.15 | 179 / 145 | 8 / 6 |
| 1,200 | 2,000 @ 0.60 | 8,000 @ 0.15 | 435 / 478 | 45 / 46 |

Doubling population at fixed adoption (2K → 4K at 0.15: 21 → 145 broadcasts,
2 → 6 warranted) moves the result to where the matched wearer count predicts, not
to where the population does.

What population buys on top of wearers is outbreak *evidence*, and that does not
match at matched wearers: 1,200 wearers gives disease TP 1 / TTD 254 at 2K but
6 / 156 at 8K. The outbreak seeds 20 people at every scale, so at 2K the disease
arm stays marginal (1 detection, 254–505 steps against 84–156 at the upper rungs)
while the plume — which raises every device in a cell at once — scales with
wearers alone. The near-universal arm below shows that 1,200 wearers is near the
outbreak's detection floor rather than a population ceiling: raise the same 2K
town to 1,700+ wearers and the disease arm detects.

The `Disease TP` column above is `detection_event_counts.disease_true_positive`.
The sibling `attributed_disease_detections` key counts something narrower and is 0
for the 0.6 arm, so the two disagree; re-derive the tables from the former.
Neither appears in `sweep_results.csv`, so per-arm single runs are needed to check
them.

### Near-universal adoption: the capability ceiling

`examples/detection_power_universal_sweep.yaml` pushes the same 2K town to
`wearable_fraction` 0.85–0.95 (same seed, gate on, `threshold_m: 8`, subsystem
adoption unchanged). This is a counterfactual, not a scenario: no population
carries instrumented wearables at that penetration, and the run reports what
observing nearly everyone would cost as well as what it would buy.

| Wearable fraction | Wearers | Broadcasts | Warranted | Disease TP | Toxin TP | Toxin TTD | Disease TTD |
|-------------------|---------|-----------|-----------|------------|----------|-----------|-------------|
| 0.60 | 1,200 | 435 | 45 | 1 | 44 | 77 | 254 |
| 0.85 | 1,700 | 765 | 81 | 8 | 73 | 80 | 120 |
| 0.90 | 1,800 | 744 | 65 | 2 | 63 | 82 | 214 |
| 0.95 | 1,900 | 870 | 86 | 14 | 72 | 77 | 96 |

- **The outbreak becomes detectable at 2K, which the ladder said it was not.**
  All six runs at 1,700+ wearers detect it (disease TP 2–14, TTD 85–216 steps,
  every one better than 0.6's 254). The 0.85 and 0.95 arms land near 10K @ 0.15's
  97 steps and 25K @ 0.15's 84, so ~1,700 wearers at 2K buys roughly what
  1,500 wearers at 10K does; the 0.90 arm is over twice as slow (see below). The
  earlier 2K result was a wearer floor rather
  than a population ceiling. Population still supplies the *number* of cases
  (the outbreak seeds 20 people at every scale, so 25K's 40 detections are out of
  reach at 2K no matter how many people are observed).
- **The plume arm is saturated.** Toxin TTD sits at 77–82 across 0.6 through
  0.95, the same floor the 25K rung reaches (75). Above ~1,200 wearers there is
  no plume latency left to buy.
- **Detection counts scatter across seeds; the 0.90 latency does not.** 0.90
  lands below 0.85 on warranted detections (65 vs 81) and disease TP (2 vs 8), and
  replicates say the *count* is a draw artifact — a second seed on the same arm
  gives 10, above the 0.85 spread. Its ~215-step disease latency, though,
  reproduces across both seeds while 0.85 gives 85–120 across three:

  | Arm | Seed | Broadcasts | Warranted | Disease TP | Toxin TTD | Disease TTD |
  |-----|------|-----------|-----------|------------|-----------|-------------|
  | 0.85 | 42 | 765 | 81 | 8 | 80 | 120 |
  | 0.85 | 7 | 692 | 67 | 8 | 80 | 104 |
  | 0.85 | 13 | 745 | 79 | 9 | 80 | 85 |
  | 0.90 | 42 | 744 | 65 | 2 | 82 | 214 |
  | 0.90 | 7 | 826 | 105 | 10 | 80 | 216 |

  Two independent seeds landing at 214 and 216 is not scatter, so read the plateau
  as flat in *whether* the outbreak is found and not flat in *when*: the 0.90 arm
  is ~2x slower than 0.85 and 0.95 for reasons this measurement does not explain,
  and it is the thing to re-measure before treating disease latency in this range
  as a smooth function of adoption.

  ```bash
  garland --config examples/detection_power_town.yaml \
    --wearable-fraction 0.85 --seed 7 --no-plots --output-dir output/univ_seed7
  ```
- **Cost per detection does not change.** Broadcasts per warranted detection are
  9.4 / 11.4 / 10.1, the same ~10 seen at every rung and every density, and
  response epsilon is one unit per broadcast. `epsilon_per_agent_per_day` stays
  bounded at 0.062–0.073 because both the numerator and the observed population
  grow, though it is not flat: the 0.95 arm is 14% above the 0.90 arm.
  `unexplained_detection_rate` rises to 0.25–0.30 from 0.26 at the ladder rungs.
- **What is left binding is channels per person, not people.**
  `mean_effective_width` is 4.787–4.790 across all three arms — identical to
  every ladder rung — because `wearable_fraction` adds observed *people* while
  `devices.adoption` decides how many channels each of them contributes. Nothing
  in this arm tests the wide end of the fleet; a fleet of 28-channel wearers is a
  different experiment from a fleet where nearly everyone carries a wristband.

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
random-walk mobility, plume onset at step 864, outbreak onset at step 1152, and
`privacy.threshold_m: 5`, `k_min: 10`, and `time_window_steps: 12`. The
plume uses release rate 200 and stability D: calibration gives a 2.66 ha
above-gate footprint, approximately 725 m downwind by 40 m crosswind, with no
grid clipping and about 10 above-gate wearables per active step at 375
wearables/km². Static mobility froze the same handful of people in the ribbon;
the committed-scale mobility probe increased distinct dosed wearables from 1
to 51. The
provenance result was measured in-session with:

```bash
garland --config examples/staged_onset.yaml \
  --detector-mode instant --n-steps 1728 --no-plots \
  --output-dir output/staged_instant_attribution
```

- **Coincidental detections:** 95.51% of disease zone-local true positives

The staged CI guards preserve this physical scale instead of shrinking only the
population. They use 2,000 agents, 60% wearable adoption, and a 1.8 km square
grid: approximately 370 wearables/km², a median of 20 above-gate wearables per
active plume step (maximum 30), and a 900 m downwind half-grid against the calibrated 725 m
tail. The guard starts the plume at step 96 for 288 steps and runs 576 steps;
this keeps both below-floor and above-floor aggregate behavior covered without
lowering the evidence floor or anomaly threshold. These concentration and
dosed-wearable counts are evaluation-only.
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

### Prior mean and first-hour onboarding

The default `BaselineTracker` starts its EMA at the channel registry's
population resting means, with `baseline_mean_prior_strength: 12` pseudo-
observations. Its early learning rate is the larger of the configured EMA
rate and `1 / (12 + t)`, so the prior protects the first observations without
preventing a device-specific resting level from being learned. Set
`baseline_mean_prior_source: zero` and a strength of `0` to reproduce the
historical zero-mean start; this comparison mode is retained because older
published figures used it.

Devices that adopt during a run (any non-`all_at_start` adoption schedule,
including its initial adopted population) suppress token emission for their
first hour via `adoption.new_device_warmup_steps: 12` at five-minute cadence.
It reuses the existing `baseline_warmup_remaining` machinery: warm-up updates
the device-local baseline but emits no anomaly tokens, and it does not spend
privacy budget. Fleet-wide `baseline_warmup_steps` still defaults to 0, so
`all_at_start` scenarios keep their historical first-hour behaviour, and an
explicit `baseline_warmup_steps` above the adoption default wins.

The covariance prior remains independently calibrated per channel. Its
`cov_sum` and `cov_counts` are plain, undecayed running sums: an early
covariance contamination therefore does not wash out automatically. Covariance
forgetting is intentionally deferred to the separate re-wear/wearer-change
reset work.

#### Prior-mean detection trade-off

The population prior changes the operating point; it is not an across-the-board
improvement. The following fixed-seed measurements compare `origin/main`
(zero-mean start) with this branch using the committed example configurations.
The cumulative broadcast and response-epsilon totals are both higher under the
population prior, while the hazard outcomes move in different directions:

| Example (six-day cumulative) | Main / zero mean | Population prior |
| --- | ---: | ---: |
| `staged_onset.yaml` broadcasts | 7,249 | 10,455 |
| `staged_onset.yaml` response epsilon | 7,249 | 10,455 |
| `staged_onset.yaml` disease time to detection | 2 steps | 12 steps |
| `staged_onset.yaml` disease true positives | 81 | 321 |
| `staged_onset.yaml` toxin time to detection | 5 steps | 1 step |
| `staged_onset.yaml` toxin true positives | 175 | 278 |
| `detection_power_town.yaml` broadcasts | 41 | 95 |
| `detection_power_town.yaml` disease time to detection | none | 230 steps |
| `detection_power_town.yaml` disease true positives | 0 | 2 |
| `detection_power_town.yaml` toxin time to detection | 101 steps | 86 steps |
| `detection_power_town.yaml` toxin true positives | 4 | 6 |

Thus broadcast volume and epsilon roughly double. In exchange,
`detection_power_town.yaml` detects a disease cluster it previously missed and
finds the toxin 15 steps sooner. In `staged_onset.yaml`, disease detection is
later (2 to 12 steps), but the old two-step result came from a covariance
distribution numbed by the zero-mean cold start rather than from reliable early
detection; the 81 to 321 true-positive change is the relevant signal.

Every previously published epsilon and detection figure in this document was
measured under the zero-mean start. Set
`baseline_mean_prior_source: zero` with zero mean-prior strength to reproduce
that historical mode; the tables above are the measured consequences of the
default population prior and are intentionally recorded separately rather than
rewriting those historical tables.

### Device-local baseline maturation

The optional `baseline_maturation` phase learns prior biometric history for
fleet-start devices only. It walks backward from `start_datetime`, synthesizes
observations, and calls `BaselineTracker.update`. This is device-local
evaluation setup, not a protocol phase: it has no protocol visibility, creates
no detection events, consumes no privacy budget, and does not touch tokens,
broadcasts, hazards, confounders, mobility, contacts, or spatial indexing.
Devices adopting during a run begin without prior maturation history and still
use the existing onboarding and `baseline_warmup_steps` machinery.

`minimum_history_days` and `maximum_history_days` configure a uniform history
when equal, or a per-device integer draw when they differ. `cadence_steps`
controls the interval between synthesized samples. Zero maximum history keeps
the phase disabled and preserves existing scenario behavior. History length
improves annual/monthly and circadian coverage at a runtime cost; coarser
cadence reduces samples while retaining broad cycle coverage.
The simulator advances hour, day-of-year, and month together using its
existing 365-day convention; this is not a real-calendar leap-year model.

Authoritative measured costs on the development box are 84.3 microseconds per
agent-step for full `observe_and_detect`, and 29.4 microseconds per
agent-sample for synthesis plus `BaselineTracker.update`. One simulated year
of learn-only history for 2,000 agents costs approximately 1.72 hours at the
native five-minute cadence, approximately 8 minutes hourly, and approximately
34 minutes at 15-minute cadence. These are sizing measurements, not protocol
performance guarantees.

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

Plume exposure uses the configured dose-derived concentration gate (default
`> 0.1`). Exposed
plume observations are classified as respiratory before the generic
multi-system fallback when they are fever-free; late-stage infection remains
febrile or multi-system because it includes a temperature increase.

## Incident detection in the complex world

Every detection measurement above stages its hazard in a deliberately
simplified world: random-walk or static mobility, a demographically flat
fleet, and no confounder engine. `examples/incident_town_college.yaml` is the
first committed scenario that stages target incidents inside the complex
world instead — the college-town archetype with five venues and schedule
mobility, the demographic fleet at `wearable_fraction: 0.85` and
`enthusiasm_sigma: 0.8`, the full chronic confounder clutter (exercise, sleep
disruption, venue crowding, sensor artifacts, background ILI), and a staged
six-day calendar:

| day | steps | staged event |
| --- | --- | --- |
| 1 | 0–287 | world settling only |
| 2 | 288–575 | civic-victory sleep-disruption wave (steps 504–540) |
| 3 | 576–863 | heat advisory (288 steps, all 60 zones) |
| 4 | 864–1151 | toxin plume release (288 steps, campus core) |
| 5 | 1152–1439 | outbreak seeding (20 index cases at 09:00, step 1260) + block fire (steps 1296–1332) |
| 6 | 1440–1727 | outbreak growth, no new events |

```bash
garland --config examples/incident_town_college.yaml --no-plots \
  --output-dir output/incident_town_college
```

The seed-42 run completed with `world_settling_status: settled` (288 settling
steps, 1,440 measured). The fleet realized 2,552 wearers among 3,000 agents
(mean 2.07 devices per wearer, 31.8% core-only, 7.7% with four or more) and a
mean effective sensing width of 5.01.

### Both staged targets are detected, on very different terms

The toxin path works essentially as designed in this world. The recorded
toxin onset — the first step any agent's concentration exceeds the exposure
gate — is step 960, a full 96 steps (8 hours) after the configured release
start at step 864: under schedule mobility the plume drifted over ground
nobody occupied until agents' schedules carried them into the footprint.
Exposure was then intermittent (85 of the 288 release steps had any gated
exposure, peaking at 42 agents in a step, 2,630 exposed agent-steps in
total). From that onset, zone-local detection was immediate
(`time_to_detection_toxin_steps: 0.0`): 116 toxin true positives, 105 of them
attributed (coincidental fraction 0.095), attributed latency 0 steps. Twenty
of the 116 toxin true positives rested on fewer than two dosed agents
(evaluation-only caveat).

The disease path, measured against a *realized* 20-case outbreak, is far
stronger than the first attempt suggested. With the seed fired at 09:00 on a
populated campus (all 20 index cases realized; the run's SEIR trace confirms
~20 infectious through day 6), the first zone-local true positive came 6
steps (30 minutes) after onset, with 212 disease true positives over the run
and 49 attributed to the outbreak at 6-step attributed latency (coincidental
fraction 0.769). Attribution still loses three of four disease detections to
co-located clutter — a zone containing the outbreak almost always also
contains exercise, crowding, or heat evidence — but the outbreak is now
detected promptly and claimed repeatedly, where the earlier mis-seeded run
managed 2 attributed detections at 408-step latency.

### Limfac: the first run staged 1 case, not 20

The first published measurement of this scenario was invalid, and the failure
mode is worth recording. The outbreak was configured for 20 index cases in a
300 m radius at step 1152 — midnight — and `_apply_outbreak_seed` truncates
silently to the candidates actually inside the radius: exactly one agent.
Every disease number in that run described a single index case that could
never grow (7-hour detection latency, 2 attributed detections, coincidental
fraction 0.955), and nothing in the summary or the log said so. Two fixes
followed. The scenario now seeds at 09:00 (step 1260, ~790 agents within
400 m of the quad), and the engine now reports `outbreak_realized_seeds`
(configured vs. realized per outbreak) in every summary and logs a warning on
truncation, so a mis-realized seed can never again masquerade as a measured
detection result.

Transmission needed the same honesty. A mobility+SEIR sweep of this world
showed that `beta` values from 0.015 down to 3e-4 all expose a quarter to
half of the town within a day of the seed — the venue and household contact
structure stacks agents at identical coordinates, so per-contact rates that
look small saturate 3,000 residents almost immediately. The committed
scenario now uses `beta: 1e-5`, the measured regime where the outbreak
doubles every 2–3 days (I: 20→~200 over ten days) instead of infecting the
town before detection could matter. Within the six-day window the signal is
therefore the 20 index cases plus early secondaries (~25 exposed by day 6) —
an early-detection problem, not a conflagration.

### The clutter is the operating condition, not noise

Of 740 detection events, 525 were warranted (328 target, 197 actionable
non-target), 37 explained, 53 artifact (rate 0.072), and 125 unexplained
(rate 0.169). The benign misattribution rate was 0.505, led by the heat wave
(157) and venue crowding (51).

The heat-advisory day is the clearest single result. Day 3 stages no target
hazard, yet produced 993 broadcasts — more than either outbreak day — and it
contaminates the channels asymmetrically: heat drives febrile and
multi-system anomalies, so the *disease* channel's single no-hazard episode
contained a false alarm (`fpr_disease: 1.0` under episode-granular counting,
one FP episode over one no-hazard episode) while the toxin channel's did not
(`fpr_toxin: 0.0`), because heat does not produce the fever-free
respiratory-dominant signature the toxin classifier requires.

Daily broadcast counts (broadcasts deferred until the alarm scales froze):
0 / 0 / 993 / 1,547 / 1,087 / 957 for days 1–6.

### Cost and background floor in the complex world

The run issued 4,584 broadcasts and 4,584 aggregate-count releases at ε 1.0
each (740 evidence releases, 1,377 no-cluster, 2,467 below-floor), for
`epsilon_per_agent_per_day: 0.255`; ten responses were suppressed for
insufficient anonymity. The settled background token rate was 0.0208 per
eligible wearable-step — against 0.0040 for the committed simple-world null
baseline. These runs differ jointly in population, mobility model, adoption,
fleet composition, and confounders, so the gap is the complex world's
aggregate clutter premium, not a decomposition into causes.

### The false-alarm floor: the complex-world null campaign

`examples/incident_town_college_null.yaml` is the same world with both target
hazards removed and nothing else changed — same geometry, venues, schedules,
fleet, and the full benign calendar (victory wave, heat advisory, block fire,
chronic confounders, background ILI). Every broadcast it issues is a false
alarm by construction. Three seeds (42/43/44):

| metric | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| total broadcasts | 4,451 | 4,685 | 4,835 |
| detection events (all target-channel, zero hazards) | 694 | 864 | 824 |
| warranted (actionable non-target) | 197 | 280 | 247 |
| explained | 84 | 73 | 82 |
| artifact | 91 | 101 | 85 |
| unexplained (rate) | 322 (0.46) | 410 (0.47) | 410 (0.50) |
| benign misattribution rate | 0.36 | 0.35 | 0.35 |
| settled background token rate | 0.0210 | 0.0206 | 0.0205 |
| ε per agent per day | 0.247 | 0.260 | 0.269 |

Daily broadcast shape is stable across seeds (0 / 0 / ~1,000 / ~1,450 /
~1,100 / ~1,080), peaking on the heat-advisory day. The operational readings:

- **The alarm load carries no incident information at this layer.** The
  incident run issued 4,584 broadcasts; the nulls issued 4,451–4,835. An
  operator watching broadcast volume alone cannot tell the hazard week from
  the hazard-free week — discrimination lives entirely in the classification
  and attribution layers, not in alarm volume.
- **The disease channel alarms constantly without disease.** Every null run
  produced 694–864 target-channel detection events and `fpr_disease: 1.0` at
  episode granularity. Heat and venue crowding drive febrile/multi-system
  anomalies indistinguishable, at this layer, from outbreak onset.
- **~46–50% of null detections are unexplained** — not attributable to any
  staged benign cause, artifact, or hazard. That is the honest noise floor of
  the current attribution layer in the complex world.
- **The floor is seed-stable.** Background token rate varies by ±1% across
  seeds; broadcast totals by ±4%. Single-seed measurements of these
  quantities are representative. Event-level outcomes are characterized
  separately below ("Seed sensitivity"): their rates replicate, their counts
  do not.

### The long detection window: attribution improves as the outbreak grows

The six-day window ends one day after seeding, so its disease measurement is
index-case-cluster detection. `examples/incident_town_college_longwindow.yaml`
is the identical scenario extended to fourteen days (4,032 steps): same world,
fleet, benign calendar, and staged incidents; days 7–14 add no new staged
events and simply let the outbreak run its doubling phase (infectious
20 → 212 by day 14) under the unchanged chronic clutter.

The per-day series comes from the new `hazard_detections_daily` summary field,
which buckets true positives and their attribution verdicts by simulated day
(zero-based, so the day-5 09:00 seed lands in bucket 4):

| day (0-based) | disease TPs | attributed | coincidental | attributed fraction |
| --- | ---: | ---: | ---: | ---: |
| 4 (seed day) | 97 | 26 | 71 | 0.27 |
| 5 | 115 | 23 | 92 | 0.20 |
| 6 | 103 | 33 | 70 | 0.32 |
| 7 | 100 | 34 | 66 | 0.34 |
| 8 | 74 | 32 | 42 | 0.43 |
| 9 | 21 | 19 | 2 | 0.90 |
| 10 | 103 | 72 | 31 | 0.70 |
| 11 | 103 | 81 | 22 | 0.79 |
| 12 | 94 | 84 | 10 | 0.89 |
| 13 | 110 | 93 | 17 | 0.85 |

(The day-9 bucket's 21 disease TPs are a one-day lull with no staged cause;
it is noted rather than explained.)

The operational readings:

- **The window was the binding limfac on attribution, not the detector.** Over
  the full fourteen days the disease coincidental fraction falls from the
  six-day run's 0.77 to 0.46, and the daily attributed fraction climbs from
  ~0.2–0.3 during the index-case cluster to ~0.8–0.9 once the outbreak has a
  few doublings behind it (days 10–13, infectious 69 → 212). As true cases
  come to dominate the febrile signal in the affected zones, attribution stops
  losing to co-located heat and venue-crowding clutter.
- **First detection is unchanged and early**: first disease true positive and
  first attributed detection both at 6 steps (30 minutes) after onset, exactly
  as in the six-day run (the attributed figure is seed-sensitive — see below);
  the toxin measurement (0-step latency, 105/116 attributed) also reproduces
  exactly, confirming the extension changed nothing before day 7.
- **The alarm load stays flat while the outbreak grows.** Days 7–14 issue
  ~900–1,250 broadcasts per day (one 1,526 outlier on day 8), a range the
  null campaign's hazard-free days also cover, even as infectious counts grow
  tenfold. Discrimination continues to live in
  attribution, not volume — an operator gets a progressively cleaner
  attributed signal, not a louder siren.
- Whole-run totals: 13,093 broadcasts, 1,462 detection events, 920 disease
  true positives (497 attributed), unexplained detection rate 0.091 (the
  denominator now includes eight extra days in which the true hazard
  dominates), ε per agent per day 0.31.

### Seed sensitivity: which event-level conclusions survive replication

Every event-level claim above rested on seed 42. Re-running the fourteen-day
scenario at seeds 43 and 44 (`--seed` overrides the file value; identical
config otherwise) establishes which conclusions are seed-stable and which
numbers were single-draw artifacts:

```bash
PYTHONPATH=src uv run --no-sync --no-build python -m garland.app \
  --config examples/incident_town_college_longwindow.yaml --seed 43 \
  --output-dir output/incident_town_college_longwindow_seed43 --no-plots
```

| metric | seed 42 | seed 43 | seed 44 |
| --- | ---: | ---: | ---: |
| realized outbreak seed | 20/20 | 20/20 | 20/20 |
| first disease TP (steps) | 6 | 3 | 4 |
| first *attributed* disease detection (steps) | 6 | 7 | 19 |
| disease TPs | 920 | 1,367 | 1,269 |
| attributed disease detections | 497 | 877 | 635 |
| disease coincidental fraction | 0.46 | 0.36 | 0.50 |
| attributed fraction, days 4–8 (cluster phase) | 0.30 | 0.34 | 0.23 |
| attributed fraction, days 10–13 (growth phase) | 0.81 | 0.88 | 0.82 |
| toxin latency (steps) | 0 | 0 | 0 |
| attributed toxin detections | 105 | 106 | 107 |
| toxin coincidental fraction | 0.09 | 0.15 | 0.12 |
| unexplained detection rate | 0.091 | 0.082 | 0.087 |
| benign misattribution rate | 0.495 | 0.487 | 0.485 |
| total broadcasts | 13,093 | 14,535 | 14,543 |
| ε per agent per day | 0.31 | 0.35 | 0.35 |
| infectious at day 14 | 212 | 317 | 247 |

The verdicts:

- **The headline finding replicates.** The cluster-phase → growth-phase
  attribution climb (~0.2–0.3 → ~0.8–0.9 daily attributed fraction) holds at
  every seed. "The window was the limfac, not the detector" is a seed-stable
  conclusion, not seed-42 luck.
- **Detection latency is seed-stable; attribution latency is not.** First
  disease TP lands at 3–6 steps (15–30 minutes) everywhere. First *attributed*
  detection ranges 6–19 steps (30–95 minutes): which zone/day the attribution
  layer first credits depends on the co-located clutter draw. Quote latency
  from the TP series; treat single-run attributed latency as a draw.
- **Counts scatter; fractions and rates do not.** Disease TP and attributed
  totals vary by ±40% across seeds, while the unexplained detection rate
  (0.082–0.091), benign misattribution rate (0.485–0.495), toxin attribution
  (105–107 attributed, 0-step latency), and ε per agent per day (0.31–0.35)
  are tight. Event *counts* from any single run are draw artifacts; the
  rates are representative.
- **The day-9 lull is reproducible, so it is a property of the world, not
  noise.** Every seed shows the same one-day collapse in disease TPs
  (21 / 30 / 18) with a near-zero coincidental count (2 / 1 / 1) in that
  bucket, i.e. what disappears is the clutter-coincident portion of the TP
  stream while attributed detections continue. The cause is still not
  established — it is seed-independent, which points at the schedule or
  confounder calendar rather than the outbreak, but that is a hypothesis and
  the bucket remains observed-not-explained.
- **Broadcast volume stays uninformative at every seed.** Totals (13.1–14.5K)
  overlap the null campaign's range scaled to fourteen days; no seed produced
  a volume signature of the growing outbreak.

Seed 43's `fnr_toxin` of 0.008 (one missed toxin instance-zone of 125) is the
only miss recorded in the three runs; disease FNR is 0.0 everywhere.

### Realized versus configured exposure for the plume and the benign calendar

Seeding was the first place configuration and realization were found to
diverge (1 case staged of 20). The same accounting now covers the plume and
the scheduled benign sources: `plume_realized_exposure` reports, per plume,
the configured window against the steps that actually dosed anyone, the first
dosed step, peak concurrent and cumulative unique dosed agents (all agents and
wearers separately); `staged_benign_realization` reports, per scheduled benign
instance, the configured window against the steps it was active, the steps it
materially affected at least one agent, and its peak and unique affected
counts. Both are seeded from configuration at model construction, so an event
that reaches nobody appears with zeroed realization instead of vanishing from
the summary. Measurement-only: the six-day scenario's headline numbers are
unchanged (4,584 broadcasts, disease TTD 6 steps, 49 attributed disease
detections, coincidental fraction 0.769, 105 attributed toxin detections).

Six-day scenario, seed 42:

| staged event | configured | realized |
| --- | --- | --- |
| `staged_plume` | 288 steps of release from step 864 | dosed anyone on 85 steps (30%), first dose step 960 (+96 steps ≈ 8 h), peak 42 concurrent (38 wearers), 1,090 unique (927 wearers) |
| `heat_0` | 288 steps from step 576 | 288 active, 288 material, 1,034 agents |
| `victory_0` | 39 steps from step 504 | 39 active, 39 material, 1,133 agents |
| `block_fire_0` | 36 steps from step 1,296 | 36 active, **0 material**, 0 agents |

Three things the numbers say that the configuration did not:

- **The plume's 8-hour onset lag is a scheduling artefact, now quantified.**
  Release runs 288 steps; only 85 of them put anyone in the footprint at a
  dose above the exposure gate. Toxin "zero-step latency" is latency measured
  from realized dose, not from release — the release-to-detection interval is
  96 steps, and that gap is the schedule, not the detector.
- **Toxin detection rests on transient exposure, not a standing cloud.** Peak
  concurrent dosed population is 42 agents (1.4% of the town) while 1,090
  distinct agents are dosed across the event: schedules sweep the population
  through the footprint. This is the mechanism behind the standing caveat that
  20 of 116 toxin true positives rest on fewer than two dosed agents.
- **The block fire was never realized.** It is active for all 36 configured
  steps and contributes 549 perturbations to 76 wearers, but no agent ever
  clears its own materiality floor (0.25), so it contributes no material
  clutter at all. Every earlier statement that the benign calendar included a
  block fire overstated the world: what the runs actually contained was a
  sub-threshold irritant whisper. The 150 m footprint at (2100, 3300) sits
  where the schedule puts almost nobody at that hour.

The last point is the operational value of this accounting: staged clutter
being known by construction is exactly why nobody checks it, and it is why an
event silently reaching nobody survived four measurement campaigns.

### Re-siting the block fire: the irritant arm made real

The unrealized block fire was re-sited rather than its materiality floor
lowered: the floor (0.25) is a calibrated threshold shared by every irritant
source, and lowering it would have promoted every sub-threshold whisper to
material clutter town-wide instead of fixing the one event that was staged in
an empty spot. A lattice scan of scheduled wearer positions during the day-5
event window found the original center (2100, 3300) materially empty — the
adjacent stadium venue only populates on Friday and Saturday evenings — while
the northeast residential blocks around (2800, 3000) hold ~480 wearers inside
the materiality radius at 12:00, rising through the afternoon, 1.4 km from
the outbreak seed at (2000, 1900). All three incident configurations (six-day,
fourteen-day, null) now stage the fire there. Runs additionally emit a run-end
warning for any configured plume or staged benign event that ends with zero
realized exposure, so this class of silent under-realization is loud rather
than a JSON field nobody reads.

Six-day scenario, seed 42, re-measured:

| quantity | before (unrealized) | after (re-sited) |
| --- | --- | --- |
| `block_fire_0` realized material steps | 0 of 36 | 36 of 36, first at step 1,296 |
| peak / unique materially affected agents | 0 / 0 | 571 / 1,056 |
| irritant-exposure contributions (agents) | 549 (76) | 25,428 (1,261) |
| total broadcasts | 4,584 | 4,687 |
| disease: attributed / coincidental / TTD | 49 / 0.769 / 6 steps | 48 / 0.767 / 6 steps |
| toxin: attributed / coincidental / TTD | 105 / 0.095 / 0 steps | 105 / 0.095 / 0 steps |

What the re-measurement says:

- **The irritant arm is now a real concurrent event, and the targets did not
  move.** A fire materially touching 1,056 residents during the outbreak's
  seeding hours changes the disease headline by one detection (49 → 48) and
  the coincidental fraction by 0.002; the toxin numbers are bit-identical.
  The deliberate overlap the calendar always claimed — a respiratory-dominant,
  fever-free event concurrent with the febrile outbreak — is finally being
  tested, and attribution keeps them apart.
- **The fire enters the attribution ledger the way clutter should.** Irritant
  exposure now appears in 28 disease-detection attributions (up from 2) and
  accounts for 25 benign misattributions (previously absent); the benign
  misattribution rate moves 0.505 → 0.531.
- **In the null world, a real fire explains detections that were previously
  unexplained.** The seed-42 null's unexplained-detection rate drops
  0.464 → 0.418 with broadcast volume flat (4,451 → 4,463): the same
  physiological disturbances occur either way, but they now trace to a staged
  cause instead of landing in the unexplained bucket.

### Device-side citizen advisories: what the person is told

Detection and attribution end at zone-level broadcasts; the advisory layer
asks what the system can honestly tell an individual. Its design constraint
is that the network must never learn a wearer's anomaly history to advise
them: the broadcast the protocol already sends carries the hazard hypothesis,
zone, and time window, and the device joins that with its own locally-held
anomaly onset ("we think you were exposed around step X") — so assembling an
advisory costs zero privacy budget and requires no contribution from the
wearer. Advisories start at tier 1 ("possible exposure — monitor, consider a
rapid panel") and sharpen only through a public channel: wearers holding an
advisory visit a clinic at an opt-in rate (0.3/day here), the clinic resolves
a diagnosis against ground truth, and the public-health side releases a
DP-noised cumulative confirmation count per (hypothesis, zone) key — released
only when the count changes, ε 0.05 per release. Every device holding a
matching advisory reads the published count and upgrades to tier 2 ("likely
exposure on day X, recommended panel") at 3 confirmations and tier 3 (adds
expected-course guidance) at 10 — contributor or not, which is the point.

Six-day incident and null scenarios, seed 42, advisories enabled:

| quantity | incident | null |
| --- | --- | --- |
| advisory precision (ever-advised who were truly exposed) | 0.476 | 0.0 |
| advisory recall (truly exposed wearables ever advised) | 0.875 | — |
| false-advisory burden (advised, never exposed) | 900 | 1,416 |
| unique agents reaching tier 1 / 2 / 3 | 1,717 / 737 / 763 | 1,416 / 0 / 0 |
| clinic visits (opt-in) | 721 | 589 |
| clinic confirmations (toxin / disease) | 199 / 8 | 0 / 0 |
| confirmation releases / added ε | 173 / 8.65 | 0 / 0.0 |

What the measurement says:

- **The advisory layer is measurement-plus-messaging, not a detector change.**
  With advisories enabled, every detection, attribution, and latency field of
  the six-day summary is bit-identical to the re-sited baseline; only the
  epsilon ledger moves (4,687.0 → 4,695.65 — the 173 confirmation releases).
- **The tier structure does exactly what it claims in the null.** 1,416
  residents of the hazard-free world receive a tier-1 "possible exposure"
  advisory over six days — that is the honest false-advisory burden of
  advising on every broadcast — but not one clinic visit confirms anything,
  so no advisory ever escalates past cautious tier-1 language. The escalated
  guidance (test panel, expected course) is gated on evidence that the null
  world cannot produce.
- **Confirmations concentrate where attribution is strong.** The plume
  produces 199 clinic confirmations against the outbreak's 8 — the same
  toxin-easy/disease-hard asymmetry every detection measurement has shown,
  now visible as how fast each hazard's advisories sharpen.
- **Non-contributors benefit by construction.** Tier upgrades come from the
  published noisy count, so the 763 agents reaching tier 3 include wearers
  who never visited a clinic; contribution buys the community sharper
  advisories, not the contributor a private benefit.

### Remaining limfacs

- The re-sited block fire is measured on the six-day scenario and the seed-42
  null; the fourteen-day results (seeds 42/43/44) and the null's seeds 43/44
  were measured with the old, unrealized siting and describe a world whose
  irritant arm was sub-threshold.
- Realized exposure is now reported for seeding, the plume, and the three
  scheduled benign sources. The chronic stochastic confounders (exercise,
  venue crowding, background ILI, sensor artifacts, onboarding) have unbounded
  instance identities and are still only summarized in aggregate by cause.
- Plume realization is counted at the model-side exposure gate, so it measures
  who was dosed, not what dose they absorbed; there is no cumulative-dose or
  concentration-integral field.
- `fpr_disease: 1.0` is episode-granular (one FP episode over one no-hazard
  episode per run); the per-broadcast false-alarm characterization above is
  what the null campaign adds.
- The six-day scenario's disease measurement remains index-case-cluster
  detection; the fourteen-day extension above is the growth-phase
  measurement. The fourteen-day arm is now replicated at seeds 42/43/44, but
  the six-day arm's event-level numbers remain single-seed (42).
- Three seeds bound the draw variation loosely, not tightly: the ±40% spread
  in event counts is a range, not a confidence interval, and the day-9 lull
  is reproducible without being explained.
- Toxin evaluation caveat stands: 20 of 116 toxin true positives rest on
  fewer than two dosed agents.
- Advisory latency is ill-posed as measured: 0.476 precision counts an agent
  as a true positive if exposure happens at any point in the run, so the
  "latency" distribution (mean −45.7 steps, min −532) is dominated by
  advisories issued on clutter anomalies *before* the wearer's first true
  exposure. A well-posed advisory-timeliness metric needs exposure-windowed
  matching, not run-scoped sets.
- The advisory tiers model message *categories*, not validated clinical
  content: nothing here is medical advice, the clinic resolves diagnoses
  against simulation ground truth (a perfect instant test that does not
  exist), and the 0.3/day opt-in visit rate is an assumption, not data.
- Clinic confirmations feed the published count but not yet the attribution
  layer; the confirmed-diagnosis-as-ground-truth-label feedback loop that
  early-phase disease attribution needs remains unbuilt.
- These are 3,000-agent runs; none of this is yet measured at city scale.
