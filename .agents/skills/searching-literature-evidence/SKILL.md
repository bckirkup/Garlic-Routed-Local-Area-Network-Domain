---
name: searching-literature-evidence
description: Search the peer-reviewed literature with the Consensus MCP server to source a GARLAND parameter — pathogen epidemiology, wearable physiology priors, sensor error, plume dispersion, mobility and occupancy — including how a hit is recorded in pathogens.json references or a channel prior. Use whenever a parameter needs a citation, or when asked what the literature says about a mechanism. Pairs with the org-level consensus-literature-retrieval skill, which owns retrieval mechanics.
---

# Searching the Literature (Consensus MCP)

## Retrieval mechanics are in the org-level skill

Load `consensus-literature-retrieval` (`~/.agents/skills/`) before searching. It
owns the tool surface, `include_full_text_chunks: true` — which is mandatory and
returns Results, Methods and tables, including for paywalled articles — query
construction, filter behaviour, result handling, and recording which section of
the paper a number was read from.

This skill is the other half: what needs sourcing in [GARLAND], and what a hit is
allowed to become here.

## GARLAND spans four literatures — search them separately

The parameters in this repo do not come from one field, and a single query will
not reach them all:

1. **Pathogen epidemiology** — `R0`, incubation and infectious periods,
   generation interval. `medical_mode=true` and `exclude_preprints=true` are
   appropriate here.
2. **Wearable physiology priors** — resting heart rate, HRV/RMSSD, respiratory
   rate, skin temperature, sleep, gait, step count. Search for
   `population distribution`, `reference values`, `age-stratified`,
   `free-living`, `consumer wearable cohort`.
3. **Sensor error** — `agreement`, `Bland-Altman`, `mean absolute percentage
   error`, `validation against ECG/polysomnography`. This is a bioengineering
   literature, not a clinical one.
4. **Hazard dispersion and mobility** — `Gaussian plume`, `Pasquill-Gifford
   stability class`, `dispersion coefficients`, `walking speed`,
   `occupancy density`, `contact rate`. Environmental and transport engineering.

Query in the vocabulary of the paper you want. `respiratory rate reference
values free-living adults wearable` finds a distribution; "normal breathing
rate" finds patient education pages.

## Filter discipline

- `medical_mode=true` restricts to ~8M top medical documents. Right for
  incubation periods, wrong for wearable-sensor validation and plume
  dispersion — it will drop the engineering journals those live in entirely.

- `human=true` and `study_types` narrow to clinical designs; sensor-validation
  and mobility studies are usually neither.

- `domain` takes academic field codes (`med`, `eng`, `env`, `cs`), not web
  domains.

- `year_min` only for "recent" asks. Pasquill-Gifford is from the 1960s and is
  still the parameterisation in use.

## Where the citation goes

This repo already has two provenance mechanisms. Use the one that fits.

**Data-driven parameters** carry `provenance` and `references` fields in
`src/garland/data/pathogens.json`:

```json
"provenance": "Incubation ~3 days and infectious ~7 days; beta calibrated to reference R0 ~8.0 using the wild-type contact-scaling factor.",
"references": [
  "Lyngse et al., Omicron transmission in Danish households (Eurosurveillance, 2022)",
  "WHO SARS-CoV-2 variant updates"
]
```

Add the DOI to the reference string when the search gives one, and say which
field each reference supports — a `references` list that covers the entry as a
whole cannot tell the next reader where `sigma` came from.

**Code-level priors** carry a comment at the definition, as the channel
cold-start priors in `src/garland/channels.py` do:

```python
# Resting heart rate, free-living adults, cohort mean 72 bpm (SD 8).
# <Author> et al. <year>, <journal> (DOI: <doi>). Grade B: general-population
# cohort standing in for the simulated shipboard/urban population.
resting_mean=72.0,
```

State **what was measured**, **in what population or setting**, the value with
its spread, and author + year + journal + DOI. Then grade it:

- **A** — direct measurement of this quantity in this setting.
- **B** — direct measurement in an analogous setting or population.
- **C** — inferred, estimated, or a declared assumption.

## Keep calibrated apart from measured

The most important distinction in this repo. `beta`, `sigma` and `gamma` are
recorded as *calibrated so `estimate_r0()` returns the reference R0* — they are
derived quantities of the SEIR implementation, not measurements. `r0`,
`incubation_days` and `infectious_days` are the measured inputs the calibration
consumes.

Consequences:

- Source the **measured** inputs and let the calibration recompute the rest.
  Never replace a calibrated rate with a literature number that happens to be
  close; the two are defined differently and the calibration is what keeps the
  entry self-consistent.
- Never relabel a calibrated value as measured because a search turned up a
  similar figure. Agreement of magnitude is not provenance.
- The channel cold-start priors are likewise measured *from this simulation's
  own benign residuals*, not from people. A literature distribution is a
  different quantity — the raw observation variance, not the mature-tracker
  residual variance. If you source one, say which of the two you have, or the
  anomaly thresholds will be tuned against the wrong spread.

## What this search must never be used for

Do not search for a value that makes a detection result come out right —
zone-local TP/FP rates, alarm calibration, or a sweep's headline. Sourcing a
parameter independently is what makes the measured detection performance a real
result; screening candidate papers by which value helps destroys that, and the
resulting number will read as evidence about the protocol when it is evidence
about the search.

Report a null result as a result: "no cohort study reports this channel's
free-living variance" is the honest route to a declared Grade C, and
`docs/SENSOR_MODALITIES.md` is where the deliberate simplification belongs.
