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
| `motion_actigraphy` | `step_count`, `sleep_fragmentation_index` | Accelerometer-only actigraph. Pedometer reports every epoch; the sleep-motion aggregate is scored once per night. |
| `instrumented_footwear` | `gait_speed_m_s`, `stride_time_variability`, `gait_asymmetry` | Shoe-borne inertial insole. Ambulation-gated: reports only while the wearer is walking. |
| `respiratory_acoustic_patch` | `cough_rate`, `speech_pause_ratio`, `adventitious_breath_fraction`, `heart_sound_s1_s2_ratio` | Adhesive chest contact microphone. Cough survives motion; heart sounds barely do. |
| `chest_electrode_patch` | `heart_rate`, `hrv_rmssd`, `qtc_ms`, `ectopy_burden`, `ptt_systolic_bp` | Adhesive two-lead ECG patch with accelerometer. Re-reports the cardiac core vitals at electrode quality, so it is partly redundancy rather than width. |

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

Actigraphy, expressed per five-minute epoch to match the rest of the vector:

| Channel | Resting mean ± between-person SD | Within-person epoch noise SD | Illness deviation | Usable duty cycle |
|---|---|---|---|---|
| `step_count` (steps/epoch, sedentary floor) | 7.5 ± 4.0 | 10.0 | −10/epoch (≈ −2,900 steps/day) | 95% |
| `sleep_fragmentation_index` (% of night restless) | 24 ± 8 | 5.0 | +12 points (febrile night) | 85% of nights |

The pedometer's resting mean is a *sedentary floor*: the ambulatory part comes
from `activity_coefficient` × activity level, so a day integrates to roughly
7,000–8,000 steps rather than the floor × 288. Sources: Tudor-Locke et al., *Int
J Behav Nutr Phys Act* (2011) for habitual step distributions and the
1,500–2,500 step/day drop during acute respiratory illness; Natale et al.,
*Chronobiol Int* (2009) and the actigraphic fragmentation-index literature for
the fragmentation baseline. Unlike the EIT/acoustic table these numbers were not
supplied as a calibration set: they are my sourcing from the actigraphy
literature, so treat them as the most revisable parameters here.

Shoe-borne gait, per walking bout rather than per calendar epoch:

| Channel | Resting mean ± between-person SD | Within-person epoch noise SD | Illness deviation | Usable duty cycle |
|---|---|---|---|---|
| `gait_speed_m_s` (habitual walking speed) | 1.30 ± 0.15 | 0.10 | −0.15 m/s (malaise) | 10% sedentary, rising to ~90% mid-bout |
| `stride_time_variability` (% CV of stride time) | 2.4 ± 0.8 | 0.6 | +2.4 points (fatigue roughly doubles CV) | 10% → ~80% |
| `gait_asymmetry` (% left-right) | 2.0 ± 1.0 | 0.8 | none — see below | 10% → ~65% |

Sources: Bohannon & Williams Andrews, *Physiotherapy* (2011) for habitual speed
and the 0.10–0.15 m/s minimal-clinically-important difference; Hausdorff,
*Hum Mov Sci* (2007) and the stride-variability/fatigue literature for the CV
baseline. Like the actigraphy table these are my own sourcing rather than a
supplied calibration set.

`gait_asymmetry` deliberately has **no** illness signature. It exists as a
negative control channel: only `instrument_artifact` moves it, so a run where
asymmetry alarms is a run where the shoe, not the wearer, changed.

Adhesive-patch acoustics, all four derived per epoch from on-node event
extraction rather than from any stored waveform:

| Channel | Resting mean ± between-person SD | Within-person epoch noise SD | Illness deviation | Usable duty cycle |
|---|---|---|---|---|
| `cough_rate` (coughs/h) | 0.8 ± 0.6 | 1.0 | +12.6 /h (respiratory infection), +18 /h (irritant) | ~88%, ~74% while active |
| `speech_pause_ratio` (% of speaking time in pauses) | 22 ± 6 | 4.0 | +9 points | ~30% sedentary awake, ~50% while active, 0% asleep |
| `adventitious_breath_fraction` (% of breaths with wheeze/crackle) | 1.5 ± 1.5 | 1.5 | +18 points | ~70%, ~82% asleep, ~20% while active |
| `heart_sound_s1_s2_ratio` | 1.15 ± 0.25 | 0.15 | +0.35 | ~60%, ~78% asleep, ~5% while active |

Sources: the cough-monitoring literature for the sub-1/h healthy waking baseline
and the order-of-magnitude rise in acute cough illness (Smith & Woodcock,
*Lung* 2006; Hall et al., *Digit Biomark* 2020); speech-pause and phrase-length
work on breathlessness for the pause ratio; auscultation-based crackle/wheeze
prevalence for the adventitious fraction; and phonocardiographic S1 amplitude's
contractility dependence for the heart-sound ratio. Like the actigraphy and gait
tables these are my own sourcing, not a supplied calibration set — the cough
baseline is the best supported and the heart-sound ratio the weakest.

Two modelling choices are worth flagging. `cough_rate` is the one channel an
irritant plume moves *harder* than an infection does, so it is deliberately not
a toxin-versus-disease discriminator; the absent `inflammatory_drive` still is.
And `adventitious_breath_fraction` is driven by both real consolidation and
instrument artifact, because a contact microphone rubbing on a shirt genuinely
cannot tell friction from a crackle — real consolidation moves it 3.6× harder,
which is what keeps the channel usable.

Two-lead chest electrodes, all per epoch:

| Channel | Resting mean ± between-person SD | Within-person epoch noise SD | Illness deviation | Usable duty cycle |
|---|---|---|---|---|
| `qtc_ms` (rate-corrected QT) | 410 ± 20 ms | 8.0 ms | +20 ms (systemic inflammation), +32 ms with enteric electrolyte loss | ~80%, ~90% asleep, ~25% while active |
| `ectopy_burden` (% of beats premature) | 0.3 ± 0.5 | 0.4 | +1.2 points | ~85%, ~58% while active |
| `ptt_systolic_bp` (cuffless systolic) | 118 ± 12 mmHg | 4.0 mmHg | +12 mmHg (sympathetic stiffening); −12 mmHg (distributive shock) | ~65%, ~80% asleep, ~15% while active |

Sources: the QT-prolongation-in-inflammation and acute-infection literature for
the QTc shift, ambulatory-monitoring ectopy prevalence for the premature-beat
baseline, and the pulse-transit-time blood pressure validation literature (which
is calibration-limited in *absolute* terms — this model only ever uses change
from a person's own baseline). Like the actigraphy, gait and acoustic tables
these are my own sourcing rather than a supplied calibration set.

Three modelling choices are worth flagging. The patch re-reports `heart_rate`
and `hrv_rmssd`, which the wrist already covers: masks are OR-ed across owned
devices, so this buys **redundancy** — an owner whose watch battery has
flattened keeps reporting rate and variability from the electrodes. `qtc_ms`
carries a negative `circadian_scale`, running longer overnight, opposite in sign
to the heart-rate circadian term it shares a driver with. And `ectopy_burden` is
the clearest case in the fleet of a channel a sensor fault moves about as hard
as an illness does — lead noise and motion transients are what beat detection
misreads as premature complexes — so it is informative only in company, when
the QT and pressure channels move with it.

Resting means and SDs, the per-epoch noise SD, and the duty cycles are wired in
as the channel definitions and device bindings. The illness effect sizes are
wired in as hazard and confounder signatures (below), using the midpoint of each
range as the fully-symptomatic magnitude. These are simulation magnitudes chosen
to match published effect sizes, not clinical claims about any device.

## Illness signatures

`garland.modality_signatures` maps illness onto latent axes rather than onto
channels directly, so one cause moves the band channels coherently instead of as
unrelated per-channel effects:

| Axis | Range | Drives |
|---|---|---|
| `inflammatory_drive` | 0…1 | `pep_ms` −27.5 ms, `gastric_emptying_index` +50 min, `heart_sound_s1_s2_ratio` +0.35, `qtc_ms` +20 ms, `ectopy_burden` +1.2 points |
| `pulmonary_involvement` | 0…1 | `regional_ventilation_heterogeneity` +0.30, `speech_pause_ratio` +9 points, `adventitious_breath_fraction` +18 points |
| `enteric_drive` | −1…1 | `bowel_sound_burst_rate` +18.5 (up) / −4.5 (ileus), `qtc_ms` +12 ms (electrolyte loss, upward drive only) |
| `arterial_stiffening` | −1…1 | `pwv_m_s` +2.15, `ptt_systolic_bp` +12 mmHg (stiffening) / both negative in distributive shock |
| `activity_withdrawal` | −1…1 | `step_count` −10/epoch (sickness behaviour) / +600 per epoch of an exercise bout |
| `sleep_disturbance` | −1…1 | `sleep_fragmentation_index` +12 points |
| `neuromotor_fatigue` | 0…1 | `stride_time_variability` +2.4 points |
| `instrument_artifact` | 0…1 | `gait_asymmetry` +2.5 points, `adventitious_breath_fraction` +5 points, `regional_ventilation_heterogeneity` +0.25, `ectopy_burden` +1.5 points |
| `airway_irritation` | 0…1 | `cough_rate` +18 /h |

`activity_withdrawal` also drives `gait_speed_m_s` (−0.15 m/s of malaise, or
+0.45 m/s while a bout is in progress), for the same reason `pwv_m_s` and
`ptt_systolic_bp` share one arterial axis: how much someone ambulates and how
fast they walk are one behavioural state, and giving them separate axes would
let one bout of sickness behaviour count twice.

The enteric path into `qtc_ms` is what lets a chest patch see a gut infection at
all: a gastroenteritis quietens the respiratory channels and still prolongs the
QT interval through electrolyte loss. Only upward `enteric_drive` does this, so
an ileus does not shorten anything.

Symptomatic infection sets `inflammatory_drive` and `arterial_stiffening` from
one value, so a fever cannot be counted twice through two vascular channels.
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
on; contact artifact runs entirely through `instrument_artifact`, so it can only
reach the channels whose transducer it actually corrupts — the impedance field,
the insole's left-right balance, a contact microphone's crackle count and an
electrode's premature-beat count — and never fakes a fever, a cough, a
heart-sound ratio or a blood pressure.

`airway_irritation` is kept separate from `pulmonary_involvement` because a
consolidated lobe can be quiet while an inhaled irritant coughs violently
without consolidating anything. Symptomatic infection holds it at 0.7 of full
scale so an irritant plume outranks it, and `incubation_axes` gives cough a
larger prodromal fraction than the febrile channels get: a dry tickly cough is
among the earliest reported symptoms.

The behavioural axes are deliberately asymmetric. Sickness behaviour removes ten
steps from an epoch; a five-minute exercise bout adds six hundred. The pedometer
is therefore a channel whose dominant confounder is two orders of magnitude
larger than its signal, and it earns its keep through the sequential detector
accumulating a sustained daily shortfall rather than through any single epoch.
Sleep and activity are also the *earliest* channels to move: `incubation_axes`
gives them a larger fraction of their symptomatic magnitude than the thermal and
vascular channels get, modelling prodromal malaise that precedes measurable
fever. A benign disrupted night (`sleep_disruption_axes`) fragments sleep and
slows the following day, so neither behavioural channel is a free illness
detector; irritant exposure and contact artifact leave both alone, preserving the
toxin-versus-disease separator.

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
  acoustics reach their best yield. It can be negative: shoes are off overnight.
- `activity_bonus` is the inverse of `activity_penalty`, for channels where
  motion is the precondition rather than the artifact. Gait yield is ~10% for a
  sedentary epoch and 65–90% mid-bout.

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
| `motion_actigraphy` | ×0.4 | ×1.0 | ×1.0 | ×0.6 |
| `instrumented_footwear` | ×0.6 | ×0.5 | ×1.0 | ×2.2 |
| `respiratory_acoustic_patch` | ×2.4 | ×0.7 | ×1.0 | ×1.4 |
| `chest_electrode_patch` | ×1.8 | ×0.8 | ×1.2 | ×1.2 |

The insole runs a cheap IMU but on a tiny cell, and shoes come off every evening
and are rarely put on a charger, which is why its removal multiplier is high and
its charge rate low. The actigraph is the cheapest subsystem here — an
accelerometer with no optical or impedance front end — and it comes off least
often, because half its signal comes from being worn overnight. The thoracic
multipliers reflect constant-current injection across 16–32
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
- **Anomaly classification gained no new categories.** Vascular,
  gastrointestinal, motor, and sleep excursions fall through to `MULTI_SYSTEM`,
  so adoption sharpens detection without changing what a zone query can ask for.
- **`sleep_fragmentation_index` is a nightly aggregate, not sleep staging.** It
  is one derived restlessness scalar reported at wake, event-gated the same way
  the gastric estimate is; no hypnogram, sleep stage, or per-epoch sleep state
  exists.
- **Steps are drawn per epoch, not accumulated.** Each epoch's count is an
  independent draw around a circadian activity profile, so there is no daily
  cumulative counter and no correlation between consecutive epochs beyond that
  profile.
- **Gait speed and stride variability disagree under exercise and agree under
  illness.** That is the intended discriminator: a bout raises speed *and*
  variability (`neuromotor_fatigue` 0.5), while malaise lowers speed and raises
  variability, so neither gait channel separates benign exertion from illness on
  its own.
- **Gait channels are ambulation-gated, not artifact-limited.**
  `DeviceChannel.activity_bonus` is the inverse of `activity_penalty`: a shoe
  cannot estimate a stride the wearer never takes, so yield *rises* with
  activity and a negative `sleep_yield_bonus` takes the overnight epochs to
  zero. There is no explicit walking-bout state — activity level stands in for
  one, so "walking" and "fidgeting energetically while seated" are the same
  thing to the model.
- **No biomechanics.** Speed, stride-time CV and asymmetry are drawn as scalars;
  no joint kinematics, ground reaction force, or stride segmentation exists.
- **No audio exists.** The patch's four channels are derived scalars: a cough
  count, a pause fraction, a labelled-breath fraction and a heart-sound
  amplitude ratio. There is no waveform, spectrogram, event timestamp, cough
  sound, or speech content anywhere in the model, and therefore no acoustic
  propagation, no beamforming and no speaker identification — which also means
  the privacy analysis cannot say anything about voice as an identifier.
- **Speech is gated by activity, not by conversation.** Activity level stands in
  for "the wearer is awake and talking", so the pause ratio is reported on the
  same statistical basis whether someone is presenting a lecture or walking
  quietly. There is no conversation state, phrase segmentation or interlocutor.
- **Cough is a rate, not a bout.** Coughing is intensely clustered in reality —
  paroxysms separated by quiet hours — while this draws an independent rate each
  epoch around the current signature, so no bout structure or post-tussive
  refractory period exists.
- **Post-perturbation floors are hard clamps.** Channels marked `hard_floor` (a
  pedometer cannot count below zero) are clamped after hazard and confounder
  deltas land, which compresses very large negative excursions rather than
  modelling the saturation properly. The older channels keep their
  synthesis-only floor, so seeded core-vitals runs are untouched.
