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

## Second-round disambiguation

Set `disambiguation.enabled: true` to enable contextual hypothesis queries.
The default is disabled, preserving committed example results. `answer_rate`
controls simulated human participation and `yes_rate` controls the approved
yes/no mix. A no-answer is not a negative and expires as unresolved.

## Benign confounders

Set `confounders.enabled: true` to exercise independent benign sources or an
optional all-zone heat wave. Contributions are cause-labelled contextual
evidence only; they do not change hazard classification or scoring.

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

### Demo runs and settlement

`quick.yaml`, `quick.toml`, `multi_hazard.yaml`, `venues.yaml`,
`pathogen_influenza.yaml`, `privacy_sweep.yaml`, and
`device_lifecycle.yaml` are demonstrations and attack/lifecycle exercises,
not settled operating points. Their runs are shorter than one 288-step
simulated day, or end at that boundary. Hazards and normal emissions can
therefore be absent or suppressed in these short runs. Baseline warm-up
suppresses dummy traffic as well as anomaly tokens, and device re-adoption
restarts local warm-up in legacy mode. Use the world-settling, fleet-cold-start,
and onboarding markers before interpreting their reported numbers.
Adoption schedules are available through the `adoption` sub-config:
`all_at_start` is the fleet cold-start case, `rollout` models a deployment
ramp, `trickle` models individual onboarding, and `cohort` models correlated
household or venue-linked onboarding. Set `initial_adopted_fraction` below
one to model newcomers entering an established population; the default
`onboarding_window_steps` is one simulated day.

### Baseline maturation

`baseline_maturation` is an optional, device-local learning phase for
fleet-start adopters. It synthesizes observations and calls
`BaselineTracker.update` before `start_datetime`; it has no protocol visibility,
does not emit detection events or consume privacy budget, and does not touch
tokens, broadcasts, hazards, confounders, mobility, contacts, or spatial state.
Set `minimum_history_days` and `maximum_history_days` equal for a uniform
history, or use a range for per-device heterogeneity. `cadence_steps` trades
fidelity against runtime. The default is zero days (disabled). Mid-run adopters
receive no prior maturation history.

`null_baseline.yaml` is a hazard-free seven-day run: it has zero initial
infection, no outbreak seeds, and `plumes: []`, so all alerts are false alarms.
`staged_onset.yaml` provides a two-day warm-in, a random-walk fleet, a
200-unit stability-D plume beginning at step 864, and an outbreak beginning at
step 1152. The plume calibration gives approximately 10 above-gate wearables
per active step at the committed 375 wearables/km² density; the mobility choice
avoids freezing one small group in the plume ribbon.

```bash
garland --config examples/null_baseline.yaml --no-plots \
  --output-dir output/null_baseline
garland --config examples/staged_onset.yaml --no-plots \
  --output-dir output/staged_onset
garland sweep --sweep-config examples/operational_detection_sweep.yaml
```

### Detection power in a mixed-modality fleet

`detection_power_town.yaml` is a 2,000-agent community with per-subsystem
adoption spanning every effective-width bucket, the drop-one-channel diagnostic
on, random-walk mobility, and the same calibrated release-200 staged plume and
outbreak as `staged_onset.yaml`. Read the
`detection_power` block of `summary.json`; `docs/OPERATIONAL_DETECTION.md`
explains what each part of it can and cannot answer.

`detection_power_adoption_sweep.yaml` is the controlled width comparison at fixed
population (core-only, one band, whole fleet), and
`detection_power_ladder_sweep.yaml` climbs 2K → 10K → 25K on the way to
city scale. Run the ladder bottom-up and stop at the first rung that misbehaves.

`detection_power_density_sweep.yaml` is the arm the ladder cannot separate:
climbing `n_agents` on a fixed grid raises residents and wearables per cell
together, so it sweeps `wearable_fraction` at fixed population instead. At the
shipped 0.15 the zone layer sees the plume but records no disease detection —
that is a wearer-density limit, not a threshold to tune.

`detection_power_universal_sweep.yaml` takes that limit to its ceiling, holding
the 2K population and pushing `wearable_fraction` to 0.85–0.95 so nearly every
resident is observed. Subsystem adoption is unchanged, so it adds observed
people rather than channels per person (`mean_effective_width` stays 4.79). It is
a counterfactual capability measurement, not an adoption forecast: at that
penetration the 2K outbreak becomes detectable in every run (disease TTD 85–216
steps across arms and seeds, against 254 at 0.6) while plume latency is already
saturated.

`heterogeneous_fleet.yaml` keeps that near-universal penetration but stops
pretending the fleet is demographically flat. Age bands come from household
composition, ownership of each kind is weighted by age affinity and by a
per-person enthusiasm factor, and infants and the very old may carry no core
device at all. Fleet-wide owner counts per kind are unchanged from
`detection_power_town.yaml`, so it isolates the *distribution* of devices across
people: 31% of wearers on the core device alone, 7.6% on four or more, and no
infant in gait shoes or a sleep headband at any adoption fraction. Read
`fleet_composition` in `summary.json`; `docs/SENSOR_MODALITIES.md` has the
per-band ownership table.

```bash
garland --config examples/detection_power_town.yaml --no-plots \
  --output-dir output/detection_power_town
garland --config examples/heterogeneous_fleet.yaml --no-plots \
  --output-dir output/heterogeneous_fleet
garland sweep --sweep-config examples/detection_power_adoption_sweep.yaml
garland sweep --sweep-config examples/detection_power_ladder_sweep.yaml
garland sweep --sweep-config examples/detection_power_density_sweep.yaml
garland sweep --sweep-config examples/detection_power_universal_sweep.yaml
```
