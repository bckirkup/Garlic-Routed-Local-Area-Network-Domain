# Garlic-Routed-Local-Area-Network-Domain

**The Privacy-Protecting Body Area Network Based Public Health Reference Architecture**

A high-performance, privacy-preserving Epidemiological Security Testbed simulation built on Mesa ABM, custom NumPy biometric synthesis (inspired by NeuroKit2 statistical principles), and OpenWearables data schema conventions.

## Overview

GARLAND simulates a town of 250,000 agents at 5-minute resolution to evaluate a decentralized, "broadcast-and-filter" differential privacy framework against co-occurring environmental hazards (airborne toxins) and infectious disease outbreaks (respiratory viruses).

The optional second-round disambiguation layer is an interpretation aid, not
validation. It asks only from protocol-visible cluster shape: narrow,
persistent, weakly confirmed clusters can raise a recent-adoption hypothesis,
while broad simultaneous activity can raise an ambient-heat hypothesis.
The former gate used model-side onboarding age, so it was oracle-validated and
made the previously published disambiguation numbers optimistic; that gate and
the `min_onboarding_wearables_in_zone` field were removed as a breaking change
to a default-off experimental feature. Simulated human approval is seeded and
is never inferred from device age or adoption metadata; devices never disclose
those fields. Reachability acknowledgements are content-free, aggregate,
noised, and subject to the `k_min` floor. Non-response is free and never
inferred as a negative; unanswered prompts expire as unresolved.

Each ask is scored model-side against the dominant active benign instance as
well-founded when its cause matches the hypothesis, unfounded when a benign
instance is present but its cause does not match, or unscored when no benign
ground truth is available for the zone. The counts obey
`well_founded + unfounded + unscored == queries issued`. Unfounded asks and
their epsilon expenditure, along with separate unscored-ask epsilon, are
reported but deliberately do not affect `discrimination_score` or hazard
metrics. Before revisiting that choice, evaluate the unfounded-ask rate under
realistic confounder mixes, the epsilon burned on unfounded asks, and whether
an unfounded ask should eventually carry a cost. This is a simulation
mechanism, not a formal DP proof or a claim of real encryption.

`examples/disambiguation_evaluation.yaml` is the mixed-benign evaluation
scenario (venue crowding, heat wave, background ILI, household onboarding
cohorts, both hypotheses enabled), measured by the operator-run
`scripts/disambiguation_ask_eval.py`, which is deliberately not part of pytest
or CI. As first measured, the follow-up query fired on 0.57 of all broadcasts
and consumed 51.2% of the run's total epsilon, with `ambient_heat` issuing 95%
of all asks at `53/660 = 8.0%` precision over scorable asks. Instrumenting the
breadth series showed why: the absolute breadth floor was calibrated against
the un-settled startup period, so the hypothesis was measuring the fleet
turning on rather than an ambient cause.

`AMBIENT_HEAT` now requires breadth sustained across windows and elevated over
a channel baseline that does not learn during world settling, and the channel
carries an explicit `ask_epsilon_budget`. Re-measured on the same scenario,
asks fall to 0.062 per broadcast and 17.6% of total epsilon, `ambient_heat`
issues 66 asks with no unfounded ask in any variant, and the budget binds with
a documented overshoot of at most one ask's cost. `recent_adoption` is
unchanged at `29/33 = 87.9%` precision with onboarding and `0/7` without it.
Full per-variant numbers, before and after, including the confounder-free
control and a tight-budget variant, are in `docs/OPERATIONAL_DETECTION.md`. The
reporting-only decision is unchanged, and unfounded asks are now 4 of 123.

### Benign confounders

The event catalogue and warrant taxonomy are described in
[`docs/EVENT_CATALOGUE.md`](docs/EVENT_CATALOGUE.md). Exposure attributes are
evaluation-only ground truth and are never placed in protocol objects or ask
content.

The disabled-by-default `confounders` sub-config generates cause-labelled
benign biometric structure without changing protocol objects or hazard
classification. Independent exercise, sleep-disruption, and sensor-artifact
sources raise the background emission rate without sharing a spatial cause.
An optional all-zone heat wave has a stable `heat_<n>` instance identity and
footprint for later scoring work. These labels are contextual evidence, not
validation, and the engine never reads SEIR or plume state. Optional
venue-crowding episodes and exogenous household background ILI are registered
as model-side benign instances for zone-local scoring experiments.

### Benign confounders

The disabled-by-default `confounders` sub-config generates cause-labelled
benign biometric structure without changing protocol objects or hazard
classification. Independent exercise, sleep-disruption, and sensor-artifact
sources raise the background emission rate without sharing a spatial cause.
An optional all-zone heat wave has a stable `heat_<n>` instance identity and
footprint for later scoring work. These labels are contextual evidence, not
validation, and the engine never reads SEIR or plume state. Optional
venue-crowding episodes and exogenous household background ILI are registered
as model-side benign instances for zone-local scoring experiments.

## Architecture

### Layer 1: CitizenAgent (Edge Device)
- Custom discrete biometric vectors (HR, HRV, RR, Core Temp) with physiologically plausible noise
- Exponential time-decay baseline: `B(t) = ∫ X(τ) · e^{-λ(t-τ)} dτ`
- Circadian + seasonal cyclical profiles with adaptive forgetting
- Mahalanobis-distance anomaly detection across multivariate parameters

### Layer 2: Hazard Engine (Confounders)
- **Infectious Disease**: SEIR compartmental model with spatial proximity transmission (calibrated to COVID-19/Influenza benchmarks)
- **Environmental Toxin**: Gaussian plume dispersion model (Pasquill-Gifford stability classes) simulating chemical leak scenarios

### Layer 3: Decentralized Privacy Protocol
- **Blind Gating**: Simulated encrypted anomaly tokens `[Zone, AnomalyType]` (plaintext in this testbed)
- **Threshold Aggregator**: Counts tokens without reading individual biometric data
- **K-Anonymity Spatial Dilution**: Expands zones to meet population threshold before broadcast
- **Reverse-Query Broadcast**: Devices in dilated zone self-evaluate
- **Uplink Perturbation**: Randomized Response + Planar Laplace location perturbation
- **Traffic Obfuscation**: Dummy noise packets from non-matching nodes

### Layer 4: Attack Simulation
- Sybil injection (false positive flooding) — `--enable-sybil`
- Deanonymization via targeted queries — `--enable-deanon`
- Correlation attacks (temporal/spatial linking)
- Eclipse attacks (message interception)
- Replay attacks (re-injection of captured tokens)

## Performance Design

- **Vectorized computation**: Agent state in flat numpy arrays; only wearable-equipped agents run biometric pipelines
- **Parameterized wearable penetration**: `wearable_fraction` (default 15%) assigned patchy by household/neighborhood
- **Device lifecycle** (optional): battery depletion, user power-off, and device removal — enable with `--enable-device-lifecycle` or `device_lifecycle.enabled` in YAML (see `examples/device_lifecycle.yaml`)
- **Hierarchical spatial index**: H3 hexagonal cells by default (~200 m at resolution 9); rectangular grid available via `--spatial-backend rect`
- **Agent mobility**: Random-walk movement each step (disable with `--static-agents`)
- **Biometric synthesis**: Fast custom NumPy by default; optional NeuroKit2 via `--biometric-synthesis neurokit` (see [docs/BIOMETRICS.md](docs/BIOMETRICS.md))
- **Adaptive forgetting**: Exponential decay kernel parameterized for privacy (configurable λ)
- **City-scale defaults**: 250,000 agents complete a 7-day run in roughly 1–3 hours on a modern CPU (see [Scaling Guide](docs/SCALING.md))

### Scaling quick start

```bash
# Fast smoke benchmark (~5K agents, <1 min)
python -m garland.benchmark --quick

# Validate your hardware at target scale before a full run
python -m garland.benchmark --n-agents 250000 --n-steps 30

# Full city run (skip plots for faster completion)
garland --n-agents 250000 --n-steps 2016 --no-plots
```

See [docs/SCALING.md](docs/SCALING.md) for memory estimates, bottleneck analysis, and privacy-vs-scale trade-offs.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Full 250K agent simulation (7 days)
garland --n-agents 250000 --n-steps 2016

# Quick test run (1000 agents, 4 hours)
garland --n-agents 1000 --n-steps 48

# With Sybil attack enabled
garland --n-agents 50000 --enable-sybil --sybil-count 30

# With correlation and eclipse attacks
garland --n-agents 5000 --n-steps 200 --enable-correlation --enable-eclipse

# With replay attack (pairs well with Sybil to seed token cache)
garland --n-agents 5000 --enable-sybil --enable-replay

# Custom privacy parameters
garland --epsilon-per-response 0.05 --k-min 100 --laplace-scale 300

# Load settings from YAML/TOML (CLI flags override file values)
garland --config examples/quick.yaml --no-plots

# Export wearable timeseries as Open Wearables JSON
garland --n-agents 1000 --n-steps 48 --export-openwearables openwearables.json

# Parameter sweep over privacy settings
garland sweep --sweep-config examples/privacy_sweep.yaml
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-agents` | 250,000 | Population size |
| `--wearable-fraction` | 0.15 | Fraction with wearable devices |
| `--enable-device-lifecycle` | off | Battery, removal, and power-off simulation |
| `--decay-lambda` | 0.01 | Baseline forgetting rate (~6.9h half-life) |
| `--baseline-warmup-steps` | 0 | Acclimate baselines before emitting anomaly tokens |
| `--threshold-m` | 5 | Anomaly count to trigger broadcast |
| `--k-min` | 50 | K-anonymity population threshold |
| `--epsilon-per-response` | 0.1 | Privacy budget per response |
| `--laplace-scale` | 200 | Geo-indistinguishability noise (meters) |
| `--seir-beta` | 0.015 | Transmission rate per contact |
| `--initial-infected` | 10 | Seed infections at start |
| `--enable-sybil` | off | Sybil false-positive flooding |
| `--enable-deanon` | off | Targeted query deanonymization |
| `--enable-correlation` | off | Temporal/spatial trajectory linking |
| `--enable-eclipse` | off | Token interception in target zones |
| `--enable-replay` | off | Stale token re-injection |
| `--sybil-count` | 20 | Fake identities per Sybil burst |
| `--attack-target-agent` | 0 | Agent index for deanon/correlation |
| `--eclipse-zones` | (target cell) | Comma-separated grid cell IDs to eclipse |

### Settlement and cold-start markers

Run summaries and sweep rows self-label whether measurements extend past the
configured `world_settling_steps` exclusion boundary. They include
`world_settling_steps`, `world_settling_complete`, `world_settling_status`,
`steps_before_world_settling`, `steps_after_world_settling`, and
`world_settling_fraction_of_run`; per-step CSV output includes
`past_world_settling`. The deprecated `background_burn_in_steps` configuration
key remains accepted as an alias. A run ending at or before the boundary has
`world_settling_status: not_settled`; only a run extending beyond it is
`settled`.

Fleet cold start is a labeled state, not an exclusion. The summary reports
`fleet_cold_start` and the full-run
`fleet_cold_baseline_wearable_step_fraction`, using the existing
`BaselineTracker.n_samples < 5` covariance-prior regime. This is a
code-defined prior-state label, not a baseline-convergence measure.
Device onboarding is also labeled:
`post_world_settling_cold_baseline_wearable_step_fraction` measures
cold-baseline wearable-steps after world settling, while
`device_re_adoption_count` and
`legacy_device_adoption_warmup_reset_count` keep lifecycle returns and legacy
warm-up resets separate. `None` indicates an undefined denominator.
First-time adoption is configurable with `adoption.mode`: `all_at_start`
expresses the fleet cold start, `rollout` is a ramp, `trickle` samples
individual adopters, and `cohort` adopts household or venue-linked groups
together. `initial_adopted_fraction` leaves an established population in
place before a non-default schedule begins. Adoption events include their step
and zone, and per-step CSV rows report `not_adopted_wearables`. The
`onboarding_window_steps` label defaults to one simulated day and is separate
from the approximately 25-minute covariance-prior regime (`n_samples < 5`).
`peak_onboarding_wearables_in_zone` measures the largest zone-local
onboarding-window population, while
`peak_onboarding_cold_wearables_in_zone` is the narrower intersection with the
covariance-prior regime.
A first-time adopter receives the configured baseline warm-up; a returning
device does not.

World settling is the only reporting exclusion. These markers are
observational and do not alter detector, aggregation, privacy, query, or
response behavior. Baseline warm-up suppresses dummy traffic as well as
anomaly tokens, and the legacy opt-in re-adoption warm-up can restart local
suppression.

## Testing

```bash
pytest tests/ -v
```

## Output

The simulation produces:
- `output/simulation_metrics.csv`: Per-step metrics (SEIR counts, detections, epsilon)
- `output/summary.json`: Summary statistics (time-to-detection, FP/FN rates)
- `output/seir_curve.png`: SEIR dynamics plot
- `output/detection_timeline.png`: Hazard onset vs. system detection
- `output/epsilon_budget.png`: Cumulative privacy expenditure
- `output/protocol_activity.png`: Token/broadcast/response activity

## Privacy Design Goals

> **Disclaimer:** GARLAND is a simulation testbed for evaluating epidemiological security architectures. It is not a certified differential privacy implementation. Tokens are plaintext structs (not homomorphic encryption), agent IDs use Python `hash()`, and no formal privacy proof accompanies the code.

The protocol is designed to explore:
1. Limiting location precision via Planar Laplace noise and K-anonymity dilution
2. Spatial zones expanded to contain ≥K agents before broadcast
3. Indicative advanced-composition-style accounting for cumulative response cost
4. Planar Laplace location perturbation, with its declared geo epsilon reported separately
5. Randomized response as a per-response local mechanism; formal repeated-query privacy is unproven

## License

Apache License 2.0 — see [LICENSE](LICENSE).
