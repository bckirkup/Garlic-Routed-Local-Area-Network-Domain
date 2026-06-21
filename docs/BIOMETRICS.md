# Biometric Synthesis in GARLAND

GARLAND generates **5-minute aggregate** biometric vectors (HR, HRV RMSSD, RR, core temperature) for wearable agents. Two synthesis backends are available.

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

Mapped types:

| Index | Open Wearables type | Unit |
|-------|---------------------|------|
| 0 | `heart_rate` | bpm |
| 1 | `heart_rate_variability_rmssd` | ms |
| 2 | `respiratory_rate` | brpm |
| 3 | `body_temperature` | °C |

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
