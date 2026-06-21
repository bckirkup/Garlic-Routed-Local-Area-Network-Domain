# GARLAND Scaling Guide

GARLAND is designed to simulate a city-scale population (default 250,000 agents) at 5-minute resolution. This document explains what scales well, what trade-offs exist, and how to estimate runtime and memory before a full run.

## Quick reference

| Population | Wearables (15%) | Init time* | Avg step time* | Init memory* | 7-day extrapolation* |
|-----------:|----------------:|---------:|---------------:|-------------:|---------------------:|
| 1,000 | 150 | ~0.01 s | ~36 ms | ~1 MB | < 1 min |
| 10,000 | 1,500 | ~0.1 s | ~400 ms | ~6 MB | ~2 min |
| 50,000 | 7,500 | ~1 s | ~2 s | ~28 MB | ~1 h |
| 100,000 | 15,000 | ~3 s | ~3.5 s | ~56 MB | ~2 h |
| 250,000 | 37,500 | ~15 s | ~1.5–4 s | ~140 MB | ~1–3 h |

\*Measured on a typical CI/cloud runner (Python 3.12, single core). Early simulation steps (before plume onset and major outbreak) sit at the low end; active hazard periods with many broadcasts sit at the high end.

Run your own numbers:

```bash
python -m garland.benchmark --quick          # fast smoke (~5K agents)
python -m garland.benchmark --n-agents 250000 --n-steps 30
```

## Architecture: what scales and what does not

### Vectorized layers (scale with total population N)

These use flat NumPy arrays and scale roughly linearly with `N`:

- **Agent positions and spatial grid** — one-time init; positions are static after setup.
- **SEIR state arrays** — compartment transitions are vectorized.
- **Plume concentration** — computed for all agents each step via masked NumPy ops.
- **Non-wearable agents** — participate in SEIR/plume only; no biometric or privacy overhead.

### Wearable-only layers (scale with W = N × wearable_fraction)

Only ~15% of agents (default) carry wearables. The privacy protocol and biometric pipeline run per wearable, not per citizen:

- Mahalanobis anomaly detection and baseline updates
- Encrypted token emission and dummy traffic
- Reverse-query responses with randomized response + planar Laplace noise

At 250K with default penetration, **W ≈ 37,500**. This is the dominant per-step cost during quiet periods.

### Protocol bursts (scale with broadcasts × zone population)

When anomaly tokens cross the aggregator threshold, the model:

1. Dilates the trigger cell until K-anonymity is met (`k_min`, default 50)
2. Broadcasts a reverse query to all wearables in the dilated zone
3. Collects and classifies responses

Step time spikes when many zones broadcast simultaneously (e.g., during a widespread outbreak or plume exposure). The implementation iterates only wearables **inside dilated zone cells**, not the full population.

### Privacy vs. scale

The **strongest privacy guarantees apply in dense, small zones** where K-anonymity dilation must expand outward to reach `k_min`. At city scale with 200 m cells over 100 km², most cells are sparse and dilation rings grow large — which is actually easier to anonymize spatially but increases broadcast fan-out.

For protocol correctness testing, prefer **small dense configs** (few thousand agents, small grid). For glamour runs, use the default 250K over 10 km × 10 km.

## Known trade-offs

### SEIR proximity cap (`max_infectious_checks`)

Transmission uses spatial proximity within a 2 m contact radius. When more than 500 agents are infectious in a single step, the engine randomly samples 500 as transmission sources. This keeps S→E contact search bounded at city scale.

- **Default:** 500 (good balance for 250K runs)
- **Increase** for higher fidelity during large outbreaks (slower)
- **Configure via** `SEIRConfig(max_infectious_checks=...)`

This is an epidemiological approximation, not a privacy limitation.

### Static agent positions

Agents do not move after initialization. This avoids rebuilding the spatial index every step and keeps cell membership cacheable. Mobility is a planned extension (README notes hex/H3 indexing for future scale-out).

### Python object overhead

Each wearable is a lightweight `CitizenAgent` with a `BaselineTracker`. At 37.5K wearables this adds ~100 MB at init but avoids reimplementing the biometric state machine in raw arrays.

## Memory model

Rough components at 250K (default 15% wearables):

| Component | Approximate size |
|-----------|-----------------|
| Position / SEIR arrays (N=250K) | ~10 MB |
| Spatial grid reverse index | ~20 MB |
| Wearable agents + baselines (W=37.5K) | ~100 MB |
| Per-step working buffers | ~10–50 MB |

Peak RSS on a full 7-day run is typically **200–400 MB** depending on metrics history and matplotlib plot generation.

## Recommendations

### Full 7-day city run (default CLI)

```bash
garland --n-agents 250000 --n-steps 2016 --no-plots
```

Expect **1–3 hours** on a modern laptop or cloud VM. Add `--no-plots` to skip matplotlib if you only need CSV/JSON output.

### Fast iteration

```bash
garland --n-agents 1000 --n-steps 48
python -m garland.benchmark --quick
pytest tests/ -v
```

### Stress-testing the privacy protocol

Use dense populations on a small grid so K-anonymity dilation is exercised:

```bash
garland --n-agents 5000 --grid-width 2000 --grid-height 2000 --k-min 100
```

## Historical note (issue #11)

Prior to the scaling pass, two bottlenecks limited city-scale runs:

1. **Python-loop position init** — replaced with vectorized NumPy assignment.
2. **Repeated `cell_of()` lookups** in nested query loops — replaced with cached per-agent cell IDs and zone-indexed wearable iteration.

Benchmarks before/after showed ~6× step-time improvement at 100K agents. The default 250K configuration is now practical for demonstration runs on commodity hardware.
