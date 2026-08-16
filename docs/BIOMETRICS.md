# Biometric Synthesis in GARLAND

GARLAND generates **5-minute aggregate** biometric vectors for wearable agents. Two synthesis backends are available.

## Channel registry

An observation is a vector of *derived per-epoch features*, one entry per
channel, ordered by the fleet's `ChannelSet` (`garland.channels`). Waveforms are
never stored or exported — a channel is whatever a device can compute on-body
and report for a five-minute epoch.

`CORE_VITALS` is the default set and reproduces the historical four-channel
layout exactly, including RNG draw order:

| Channel | Unit | System |
|---------|------|--------|
| `heart_rate` | bpm | cardiac |
| `hrv_rmssd` | ms | cardiac |
| `respiratory_rate` | brpm | respiratory |
| `body_temperature` | °C | thermal |

Each `Channel` carries the parameters its consumers need: population resting
distribution, circadian/seasonal/activity coefficients, noise, optional floor,
excursion and quiet thresholds, covariance prior, and Open Wearables type.
Profiles, baselines, sequential detectors, hazard and confounder deltas,
anomaly classification, and export all address entries by channel *name*, so a
wider set requires no changes to those consumers:

```python
from garland.channels import CORE_VITALS

wide = CORE_VITALS.with_channels(SYSTOLIC_BP)
wide.delta({"heart_rate": 15.0})  # named perturbation, zero elsewhere
```

Classification rules are written against physiological *systems*
(`ChannelSystem`) and each channel's own `deviation_threshold`, so adding a
second cardiac or respiratory channel does not require new rules. A signature
naming a channel the fleet does not carry raises rather than silently dropping
the effect.

Anomaly thresholds are not yet degrees-of-freedom aware: the Mahalanobis cut is
calibrated for the four-channel default, and a wider set changes the null tail.
Calibrated thresholds and masked scoring for missing channels are follow-up
work.

## When to use each backend

| Backend | Default | Speed | Use case |
|---------|---------|-------|----------|
| **`custom`** | Yes | Fast (NumPy) | City-scale runs (250K agents), CI, production simulations |
| **`neurokit`** | No | Slow (~0.3 s/obs) | Validation, research subsets, comparing statistical properties against ECG/RSP simulation |

### Custom synthesis (default)

- Direct NumPy draws from physiologically plausible distributions
- Circadian and seasonal modulation on HR, RR, and temperature
- No continuous waveform storage — only aggregate vectors per 5-minute step
- Used automatically unless `--biometric-synthesis neurokit` is set

### Adaptive baseline tracking

`BaselineTracker` uses an EMA with the configured `decay_lambda` and stores
circadian/monthly patterns as deviations from that EMA. An unseen cyclical bin
therefore contributes no correction rather than pulling the expected baseline
toward zero. The default EMA rate, `0.01` per five-minute step, has a
half-life of approximately 69 steps (5.8 hours). The default cyclical learning
rate, `0.001`, has a half-life of approximately 693 updates; elapsed wall-clock
time depends on how often a bin is visited.

Covariance is accumulated from the same pre-update residual used for scoring
and combined with an explicit prior. This keeps the centre and covariance
calibrated to the same residual process during adaptation.

### NeuroKit2 synthesis (optional)

- Simulates ECG and respiratory signals via [NeuroKit2](https://neuropsychology.github.io/NeuroKit/)
- Extracts heart rate, HRV (RMSSD), and respiratory rate from processed signals
- Core temperature still uses the custom circadian model (NeuroKit2 does not provide body-temperature simulation in this path)
- Requires optional dependencies: `pip install -e ".[biosignals]"`

**Not recommended** for large populations. A 1,000-agent run with 15% wearables and NeuroKit2 synthesis can take hours per step. Reserve for small validation runs (e.g. `--n-agents 50 --biometric-synthesis neurokit`).

Configure the simulation window with `--neurokit-window 60` (seconds; default 60).

## Open Wearables export

Observations can be exported in the [Open Wearables](https://openwearables.io/docs/architecture/data-types) timeseries format:

```python
from datetime import datetime, timezone
from garland.openwearables import export_timeseries_payload, observation_to_records

records = observation_to_records(obs_vector, datetime.now(timezone.utc))
payload = export_timeseries_payload(records, resolution="5min")
```

Or from the CLI after a simulation run (timestamps use `start_datetime` + 5 minutes per step):

```bash
garland --n-agents 1000 --n-steps 48 \
  --export-openwearables openwearables.json \
  --openwearables-max-agents 10
```

Relative paths are written under `--output-dir` (default `output/`). Absolute paths are used as given. Use `--openwearables-max-agents` to cap export size on large runs.

Mapped types come from each channel's `openwearables_type`; channels with no
equivalent in that schema are omitted from exports.

| Channel | Open Wearables type | Unit |
|---------|---------------------|------|
| `heart_rate` | `heart_rate` | bpm |
| `hrv_rmssd` | `heart_rate_variability_rmssd` | ms |
| `respiratory_rate` | `respiratory_rate` | brpm |
| `body_temperature` | `body_temperature` | °C |

## CLI examples

```bash
# Default: fast custom synthesis
garland --n-agents 5000 --n-steps 48

# NeuroKit2 validation run (install biosignals extra first)
pip install -e ".[biosignals]"
garland --n-agents 50 --n-steps 10 --biometric-synthesis neurokit
```

## Configuration file

```yaml
n_agents: 100
biometric_synthesis: neurokit
neurokit_window_seconds: 60
```
