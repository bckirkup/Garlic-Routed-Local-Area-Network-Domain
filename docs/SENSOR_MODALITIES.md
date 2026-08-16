# Sensor Modalities and Device Bundles

GARLAND models a mixed-modality wearable fleet: people adopt *devices*, and each
device reports a bundle of derived scalar channels. This document records what is
simulated, the calibration the parameters come from, and the modelling
simplifications that are deliberate.

GARLAND is a simulation testbed. Nothing here is a claim of research-grade
hardware validity, and no waveform is ever synthesised or stored — a channel is
one derived feature per five-minute epoch, which is what an on-node DSP would
stream after demodulation and event extraction.

## Ownership is device-level

`garland.devices` makes the device the unit of both ownership and missingness:

- `DeviceKind` — one wearable a person can own, carrying a tuple of
  `DeviceChannel` bindings (a channel plus the yield that hardware achieves for
  it in the field).
- `DeviceFleet` — assigns ownership across the wearable population and, each
  epoch, produces the observed-channel mask the detectors consume.

The fleet keeps **one** observation layout for the whole population: the union of
the channels of every adopted kind, core vitals first. Someone who does not own
the abdominal band simply has those channels permanently missing, which is the
mask semantics `BaselineTracker` already implements — missing channels are
marginalized out of the Mahalanobis score, learn nothing, and, because the cut is
re-expressed at the reported width (`garland.thresholds`), do not change that
person's alarm rate. Mixed ownership therefore costs no false positives, which is
the property that makes modality-adoption sweeps measurable at all.

Two distinct kinds of missingness follow from this:

- **Structural** — no device, so the channel is missing in every epoch.
- **Duty cycle** — the device is owned but returns nothing usable this epoch,
  driven by motion, contact quality, sleep/wake state, or an event window.

## Device catalogue

| Kind | Channels | Notes |
|---|---|---|
| `wrist_ppg` | `heart_rate`, `hrv_rmssd`, `respiratory_rate`, `body_temperature` | The historical GARLAND device; always present, reports every epoch. |
| `thoracic_eit_acoustic_band` | `regional_ventilation_heterogeneity`, `pep_ms`, `pwv_m_s` | Multi-frequency EIT plus multipoint contact acoustics. |
| `abdominal_acoustic_band` | `bowel_sound_burst_rate`, `gastric_emptying_index` | Contact microphones plus impedance; gastric estimate is event-gated. |

Enable modalities from a config file:

```yaml
devices:
  enabled: true
  adoption:
    thoracic_eit_acoustic_band: 0.05
    abdominal_acoustic_band: 0.02
```

or from the CLI, repeating the flag per kind:

```bash
garland --device-adoption thoracic_eit_acoustic_band=0.05 \
        --device-adoption abdominal_acoustic_band=0.02
```

Adoption fractions are sampled independently per kind, so owning the thoracic
band tells you nothing about owning the abdominal one.

## Calibration

Resting distributions, within-person epoch noise, illness effect sizes, and
usable duty cycles for the EIT/contact-acoustic channels:

| Channel | Resting mean ± between-person SD | Within-person epoch noise SD | Illness deviation | Usable duty cycle |
|---|---|---|---|---|
| `regional_ventilation_heterogeneity` (EIT Global Inhomogeneity index) | 0.40 ± 0.06 | 0.025 | +0.20 to +0.40 (pneumonia, atelectasis) | 70–80% |
| `bowel_sound_burst_rate` (bursts/min) | 5.5 ± 2.0 | 1.8 | +12 to +25 (enteritis); −4.5 (ileus) | 45–60% |
| `pep_ms` (pre-ejection period) | 102 ± 14 ms | 5.0 ms | −20 to −35 ms (febrile ILI, inotropy) | 65–80% |
| `pwv_m_s` (central pulse wave velocity) | 7.2 ± 1.2 m/s | 0.4 m/s | +1.5 to +2.8 m/s (inflammation); −1.5 to −2.5 m/s (distributive shock) | 60–75% |
| `gastric_emptying_index` (liquid T½, min) | 45 ± 12 min | 6.0 min | +35 to +65 min (systemic infection) | 50–65%, postprandial windows only |

Sources: Zhao et al., *Crit Care* (2009); Frerichs et al., *Thorax* (2017);
Ozawa et al., *Clin Neurophysiol* (2010); Wang et al., *IEEE TBME* (2019);
Berntson et al., *Psychophysiology* (2004); Bosch et al., *PLoS ONE* (2018);
Reference Values for Arterial Stiffness Collaboration, *Eur Heart J* (2010);
Couturier et al., *Am J Physiol* (2001); Cremonini et al., *Gut* (2002).

Resting means and SDs, the per-epoch noise SD, and the duty cycles are wired in
as the channel definitions and device bindings. The illness effect sizes are
wired in as hazard and confounder signatures (below), using the midpoint of each
range as the fully-symptomatic magnitude. These are simulation magnitudes chosen
to match published effect sizes, not clinical claims about any device.

## Illness signatures

`garland.modality_signatures` maps illness onto four latent axes rather than onto
channels directly, so one cause moves the band channels coherently instead of as
unrelated per-channel effects:

| Axis | Range | Drives |
|---|---|---|
| `inflammatory_drive` | 0…1 | `pep_ms` −27.5 ms, `gastric_emptying_index` +50 min |
| `pulmonary_involvement` | 0…1 | `regional_ventilation_heterogeneity` +0.30 |
| `enteric_drive` | −1…1 | `bowel_sound_burst_rate` +18.5 (up) / −4.5 (ileus) |
| `arterial_stiffening` | −1…1 | `pwv_m_s` +2.15 (stiffening) / −2.15 (distributive shock) |

Symptomatic infection sets `inflammatory_drive` and `arterial_stiffening` from
one value, so a fever cannot be counted twice through two vascular channels, and
a future pulse-transit-time blood pressure channel reads the same arterial axis.
A pathogen's `enteric_involvement` (0 = purely respiratory, 1 =
gastroenteritis-dominant; 0.9 for norovirus) shifts a fixed amount of tropism
from `pulmonary_involvement` to `enteric_drive` without changing how systemically
ill the agent is — so route-of-transmission structure shows up in *which* band
fires, not in severity.

Confounders share the machinery, which is the point: background ILI carries the
same axes at 0.6 severity, and exercise shortens PEP and stiffens arteries, so
the bands are not a free discriminator between benign and outbreak illness.
Irritant plume exposure raises ventilation heterogeneity with no inflammatory
drive, keeping the toxin-versus-disease separation the classifier already relied
on; contact artifact corrupts the impedance field only.

Deltas are emitted only for channels present in the running set, so a core-vitals
fleet is unaffected and adopting one band never perturbs the other's channels.

## Duty cycle model

`DeviceChannel.yield_probability` starts from the channel's duty cycle and
applies the two effects the calibration attributes the losses to:

- `activity_penalty` scales with the wearer's activity level. Gross motion is
  what actually destroys these channels: electrode contact drift for impedance,
  fiducial-point loss for cardiac timing, garment friction and speech for
  contact acoustics.
- `sleep_yield_bonus` applies during the 22:00–06:00 window, where abdominal
  acoustics reach their best yield.

`event_completion_hours` marks a channel as event-gated instead: a liquid-meal
T½ is integrated over a two-to-three hour postprandial window, so
`gastric_emptying_index` reports in the epochs containing configured completion
times (three per day) rather than being resampled every epoch.

## Per-subsystem batteries

Each device kind is separate hardware, so it gets its own lifecycle engine:
`SubsystemLifecycle` runs one `DeviceLifecycleEngine` per adopted kind, keyed by
kind name. A flat or removed thoracic band masks only its own three channels;
the watch on the same wrist keeps reporting, and vice versa. Charging recovers
only the subsystem that ran down.

One `device_lifecycle` config block still drives the whole fleet, scaled per kind
by `SubsystemPowerProfile`, so the ordering between subsystems survives tuning:

| Kind | Drain | Capacity | Activity drain | Removal |
|---|---|---|---|---|
| `wrist_ppg` | ×1.0 | ×1.0 | ×1.0 | ×1.0 |
| `thoracic_eit_acoustic_band` | ×3.0 | ×1.6 | ×1.5 | ×2.0 |
| `abdominal_acoustic_band` | ×2.0 | ×1.3 | ×1.0 | ×2.5 |

The thoracic multipliers reflect constant-current injection across 16–32
electrodes cycled to 1 MHz plus synchronous multi-channel acoustic sampling,
against a larger torso cell; the removal multipliers reflect that a band comes
off for showers and sleep more readily than a watch does. These are ordering
assumptions about a hypothetical device, not measured battery lives.

The wrist device keeps the historical per-person path: its status still lives on
`CitizenAgent.device_status`, and an agent who has not adopted at all reports
nothing from any subsystem. Per-subsystem active counts, depleted/not-worn
counts, and mean battery appear in the metrics CSV as
`subsystem_<kind>_active` and friends, alongside the unchanged `wearables_*`
wrist fields.

## Known simplifications

- **Yield draws are independent per channel.** Real artifact is correlated within
  a band — one torso twist spoils every frame at once. Modelling band-level
  correlated dropout is future work.
- **`pwv_m_s` is an independent channel.** It shares its underlying arterial
  stiffness state with a future pulse-transit-time blood pressure channel; until
  that latent state exists, adding both would double-count one physiological
  change in the Mahalanobis score.
- **Subsystem lifecycles are independent, including their failures.** Nothing
  correlates a band coming off with the watch coming off, though in practice a
  person who stops wearing one device is more likely to stop wearing another.
- **Illness signatures are linear in severity.** Each axis scales its channel
  deltas linearly with symptom progress; real trajectories (consolidation,
  ileus) are neither linear nor reversible on the same timescale.
- **Anomaly classification gained no new categories.** Vascular and
  gastrointestinal excursions fall through to `MULTI_SYSTEM`, so band adoption
  sharpens detection without changing what a zone query can ask for.
