# Garlic-Routed-Local-Area-Network-Domain

**The Privacy-Protecting Body Area Network Based Public Health Reference Architecture**

A high-performance, privacy-preserving Epidemiological Security Testbed simulation built on Mesa ABM, custom NumPy biometric synthesis (inspired by NeuroKit2 statistical principles), and OpenWearables data schema conventions.

## Overview

GARLAND simulates a town of 250,000 agents at 5-minute resolution to evaluate a decentralized, "broadcast-and-filter" differential privacy framework against co-occurring environmental hazards (airborne toxins) and infectious disease outbreaks (respiratory viruses).

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
- **Uplink Perturbation**: Randomized Response + Planar Laplace geo-indistinguishability
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
3. Adaptive composition accounting for cumulative privacy loss: `ε_total ≈ ε√(2n·ln(1/δ))`
4. Planar Laplace mechanism for approximate geo-indistinguishability
5. Randomized response for plausible deniability

## License

Apache License 2.0 — see [LICENSE](LICENSE).
