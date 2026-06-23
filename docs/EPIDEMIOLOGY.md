# Epidemiological Parameters

GARLAND's SEIR hazard engine uses compartmental rates (`beta`, `sigma`, `gamma`) tuned for privacy-protocol testing at 5-minute resolution (288 steps per day). The bundled **pathogen library** (`src/garland/data/pathogens.json`) documents literature-grounded profiles so runs can swap disease dynamics without hand-editing rates.

## Default provenance

The built-in `SEIRConfig` defaults match the `covid19_wildtype` entry:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `beta` | 0.015 | Per-contact transmission probability per 5-min step within `contact_radius` |
| `sigma` | 0.000694 | E→I rate (≈ 5-day incubation) |
| `gamma` | 0.000347 | I→R rate (≈ 10-day infectious period) |
| `contact_radius` | 2.0 m | Proximity contact search radius |

`sigma` and `gamma` are computed as `1 / (period_days × 288)`. `beta` values in the library are calibrated from literature R₀ using the wild-type reference scale factor (see `beta_from_r0()` in `garland.pathogens`).

## Pathogen library

Available pathogen ids (see JSON for full metadata):

| Id | Reference R₀ | Incubation | Infectious period |
|----|--------------|------------|-------------------|
| `covid19_wildtype` | 2.5 | 5 d | 10 d |
| `covid19_omicron` | 8.0 | 3 d | 7 d |
| `influenza_seasonal` | 1.3 | 2 d | 5 d |
| `measles` | 15.0 | 10 d | 8 d |
| `rsv` | 3.0 | 4 d | 7 d |
| `norovirus` | 2.0 | 1.5 d | 3 d |

Each entry includes `provenance`, `references`, SEIR rates, and suggested outbreak seed counts.

## Loading via config

Set `seir.pathogen` in YAML or TOML. Explicit SEIR fields override library defaults:

```yaml
seir:
  pathogen: covid19_omicron
  initial_infected: 20   # overrides the pathogen default seed count
```

List ids from Python:

```python
from garland.pathogens import list_pathogen_ids, get_pathogen

print(list_pathogen_ids())
profile = get_pathogen("influenza_seasonal")
```

## R₀ estimation helper

For regression checks and calibration scripts:

```python
from garland.pathogens import beta_from_r0, estimate_r0

r0 = estimate_r0(beta=0.015, gamma=0.000347)
beta = beta_from_r0(target_r0=3.0, gamma=0.000496)
```

This uses the wild-type reference profile as the contact-scaling anchor.

## Related work (issue #49)

The pathogen library is the first step toward full SEIR calibration against observed incidence curves. Future work may add fitting helpers that write calibrated entries back into run configs.
