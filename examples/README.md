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
