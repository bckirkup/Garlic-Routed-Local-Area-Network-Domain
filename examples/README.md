# Example GARLAND Config Files

These files configure a single simulation run. Load one with:

```bash
garland --config examples/quick.yaml --no-plots
garland --config examples/quick.toml --no-plots
```

CLI flags override file values when they differ from defaults:

```bash
garland --config examples/quick.yaml --n-agents 500 --epsilon-per-response 0.05
```

## Parameter sweeps

Use `garland sweep` with a sweep definition:

```bash
garland sweep --sweep-config examples/privacy_sweep.yaml
```

Sweep configs support either:

- **`sweep`**: Cartesian product over dotted parameter paths
- **`runs`**: Explicit list of named runs with nested overrides

Results are written to `output/privacy_sweep/sweep_results.csv` (or the configured `output_dir`).

## Structured venues

`venues.yaml` demonstrates schedule-driven mobility with calibrated activity
patterns (work, school, hospital, shopping, third places) and elevated
venue-local SEIR transmission:

```bash
garland --config examples/venues.yaml --no-plots
```

Use `venues.calibration_preset` (`us_urban_weekday`, `us_suburban`,
`weekend_leisure`) or override `venues.calibration` dwell curves to match
cell-phone stay-point data for your region.

## Pathogen library

`pathogen_influenza.yaml` loads SEIR parameters from the bundled pathogen
library via `seir.pathogen`:

```bash
garland --config examples/pathogen_influenza.yaml --no-plots
```

See `docs/EPIDEMIOLOGY.md` for available pathogen ids and parameter provenance.

## Operational detection scenarios

### Demo runs and burn-in

`quick.yaml`, `quick.toml`, `multi_hazard.yaml`, `venues.yaml`,
`pathogen_influenza.yaml`, `privacy_sweep.yaml`, and
`device_lifecycle.yaml` are demonstrations and attack/lifecycle exercises,
not settled operating points. Their runs are shorter than one 288-step
simulated day, or end at that boundary. Hazards and normal emissions can
therefore be absent or suppressed in these short runs. Baseline warm-up
suppresses dummy traffic as well as anomaly tokens, and device re-adoption
restarts local warm-up. Use the run's burn-in markers before interpreting
their reported numbers.

`null_baseline.yaml` is a hazard-free seven-day run: it has zero initial
infection, no outbreak seeds, and `plumes: []`, so all alerts are false alarms.
`staged_onset.yaml` provides a two-day warm-in, a plume beginning at step 864,
and an outbreak beginning at step 1152.

```bash
garland --config examples/null_baseline.yaml --no-plots \
  --output-dir output/null_baseline
garland --config examples/staged_onset.yaml --no-plots \
  --output-dir output/staged_onset
garland sweep --sweep-config examples/operational_detection_sweep.yaml
```
