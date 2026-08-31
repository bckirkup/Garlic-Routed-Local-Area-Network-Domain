# Changelog

All notable changes to GARLAND are documented here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Added an optional host-phenotype layer for diabetic, frail-elderly,
  law-enforcement, and assistive-need populations, including host-dependent
  susceptibility, illness presentation, confounder alignment, and need-gated
  hearable ownership.
- Added wrist SpO2, distal skin-temperature, and EDA channels, with expanded
  wrist and EDA-module device bundles for the wearable fleet.
- Device-side citizen advisories (`src/garland/advisories.py`, opt-in via the
  new `advisories:` config block, enabled in the six-day incident and null
  scenarios): each wearable assembles its own advisory locally by joining a
  received broadcast's hazard hypothesis with its device-local anomaly onset,
  so no network party learns individual history and non-contributors benefit
  identically. Advisories start at tier 1 ("possible exposure") and upgrade
  to tiers 2–3 (test-panel and expected-course message categories) from a
  DP-noised public confirmation count fed by opt-in clinic visits, released
  only when the count changes and epsilon-accounted into the aggregator total.
  Measurement-only metrics land under `advisories` in the summary: precision,
  recall, latency, false-advisory burden, tier distribution, clinic visits,
  confirmations by type, and release epsilon. Six-day seed-42 measurement
  ("Device-side citizen advisories" in `docs/OPERATIONAL_DETECTION.md`):
  precision 0.476 / recall 0.875, 763 agents reach tier 3 in the incident
  world for 8.65 added epsilon, while the null world's 1,416 false advisories
  never escalate past tier 1 (zero confirmations, zero releases) and the
  detection pipeline is bit-identical with the layer on or off.
- Re-sited the staged block fire in all three incident configurations from the
  materially empty (2100, 3300) to the northeast residential blocks at
  (2800, 3000), where the day-5 schedule puts ~480 wearers inside the
  materiality radius at 12:00, 1.4 km from the outbreak seed. The irritant arm
  is now realized: 36 of 36 material steps, peak 571 / unique 1,056 affected
  agents, irritant contributions 549 → 25,428 — while the target headlines are
  effectively unchanged (disease 48 attributed / 0.767 coincidental / 6-step
  TTD; toxin bit-identical) and the seed-42 null's unexplained-detection rate
  drops 0.464 → 0.418 as previously unexplained detections trace to the now
  real fire ("Re-siting the block fire" in `docs/OPERATIONAL_DETECTION.md`).
- Runs now emit a run-end warning for every configured plume with zero
  realized exposed steps and every scheduled benign event with zero realized
  material steps, so a configured event that reaches nobody is loud at the
  console instead of a zeroed field in the summary JSON.
- Summaries now report configured-versus-realized exposure for every staged
  event, not just outbreak seeding: `plume_realized_exposure` (per plume:
  configured window, dosed steps, first dosed step and onset lag, peak
  concurrent and cumulative unique dosed agents/wearers) and
  `staged_benign_realization` (per scheduled benign instance: configured
  window, active steps, material steps, first material step, peak and unique
  affected agents). Both are seeded from configuration at model construction,
  so an event that reaches nobody appears with zeroed realization instead of
  vanishing. First measurement ("Realized versus configured exposure" in
  `docs/OPERATIONAL_DETECTION.md`): the plume's 8-hour onset lag is 96 steps
  of empty footprint quantified (85 of 288 release steps dosed anyone; peak
  42 concurrent vs 1,090 unique dosed), and the staged block fire turns out
  never to have been realized — 36 active steps, zero agents above its
  materiality floor — so the benign calendar's irritant arm was sub-threshold
  in every campaign to date.
- Added `examples/incident_town_college_longwindow.yaml`, the fourteen-day
  extension of the complex-world incident scenario: identical world, fleet,
  benign calendar, and staged incidents, with days 7–14 observing the
  outbreak's doubling phase (infectious 20 → 212). The seed-42 measurement
  ("The long detection window" in `docs/OPERATIONAL_DETECTION.md`) shows the
  short window was the binding limfac on attribution: the disease
  coincidental fraction falls from 0.77 (six-day run) to 0.46, with the daily
  attributed fraction climbing from ~0.2–0.3 during the index-case cluster to
  ~0.8–0.9 in the growth phase, while daily broadcast volume stays inside the
  hazard-free null range throughout.
- Documented the three-seed replication of the fourteen-day arm ("Seed
  sensitivity" in `docs/OPERATIONAL_DETECTION.md`): the cluster-phase →
  growth-phase attribution climb reproduces at seeds 42/43/44, and rates
  (unexplained detection 0.082–0.091, benign misattribution 0.485–0.495,
  toxin 0-step latency with 105–107 attributed) are tight while event counts
  scatter by ±40%. First disease true positive is seed-stable at 3–6 steps,
  but first *attributed* detection ranges 6–19 steps, so single-run
  attributed latency is a draw rather than a measurement.
- Summaries now report `hazard_detections_daily`: per-day, per-hazard counts
  of true positives split into attributed and coincidental, so growth-phase
  attribution can be read from a single run instead of whole-run totals.
- The per-day CLI progress line now flushes immediately, so redirected runs
  show interim progress instead of buffering it until exit.
- Added `examples/incident_town_college.yaml`, the first committed scenario
  that stages target incidents inside the complex world: the college-town
  archetype with five venues and schedule mobility, the demographic fleet at
  0.85 adoption, the full chronic confounder clutter, and a six-day staged
  calendar (civic-victory wave, heat advisory, toxin plume, outbreak seeding
  plus block fire, outbreak growth). The seed-42 measurement is recorded in
  `docs/OPERATIONAL_DETECTION.md` ("Incident detection in the complex
  world"): both staged targets are detected zone-locally (toxin attributed
  fraction 0.905 with zero-step latency from first gated exposure; the
  realized 20-case outbreak detected 30 minutes after onset with 49
  attributed detections, coincidental fraction 0.769), the hazard-free
  heat-advisory day produces more broadcasts than either outbreak day and
  contaminates the disease channel but not the toxin channel, and the
  settled background token rate runs roughly five times the simple-world
  null baseline. The outbreak seed fires at a populated hour with
  transmission calibrated (`beta: 1e-5`) to the world's dense venue contact
  structure so it doubles every 2–3 days instead of saturating the town in a
  day.
- Added `examples/incident_town_college_null.yaml`, the complex-world
  false-alarm campaign: the same world with both target hazards removed, so
  every broadcast is a false alarm by construction. Across seeds 42/43/44 it
  issues 4,451–4,835 broadcasts and 694–864 target-channel detection events
  per six-day run — statistically indistinguishable in volume from the
  incident run — with ~46–50% of detections unexplained and a seed-stable
  settled background token rate of 0.0205–0.0210 (measurements in the same
  docs section).
- Outbreak seeding is now observable: every summary reports
  `outbreak_realized_seeds` (configured vs. realized index cases per
  outbreak) and the engine logs a warning when a seed truncates to the
  agents actually inside its radius, after a mis-realized seed (1 case
  realized of 20 configured, seeded at midnight into an empty campus)
  silently invalidated the scenario's first disease measurement.
- Added an opt-in `demographics` block (`garland.demographics`) giving the fleet
  an age structure and making device ownership correlated within a person and
  conditioned on age. Age bands (infant, child, adult, older adult, elderly)
  come from household composition rather than independent per-person draws, so
  juveniles co-reside with adults and seniors cluster; ownership of each kind is
  then weighted by an age affinity, where zero is a hard exclusion (no infant
  gait shoes or sleep headbands at any adoption fraction), times a mean-one
  lognormal per-person enthusiasm factor with spread `enthusiasm_sigma`.
  Per-band `base_device_retention` lets an infant or a very old person in an
  adopting household carry no core device, with the wearable population topped
  up so `wearable_fraction` still holds within a few percent. Owner counts per
  kind are unchanged — mean devices per wearer is 2.069 at every sigma — so
  every committed adoption fraction keeps its meaning while the distribution
  across people becomes long-tailed: at the town operating point the core-only
  share runs 25.2% → 38.3% and the four-or-more share 4.5% → 10.5% across sigma
  0.0 → 1.6. Runs now report `fleet_composition` (age bands in the population
  and among wearers, owners per kind, owners per kind per band, and the spread
  of device counts), and `examples/heterogeneous_fleet.yaml` is the scenario at
  town scale. Defaults are unchanged and disabled: the fleet stays
  demographically flat with independent per-kind draws. Age drives ownership
  only — physiology remains age-blind, so per-band anomaly rates are not
  clinical age effects.
- Added a calibrated population prior mean with configurable pseudo-count
  strength for device baselines, preserving a selectable zero-mean mode for
  historical comparisons. Devices adopting during a run now default to a
  one-hour baseline warm-up suppression via
  `adoption.new_device_warmup_steps`; covariance sums remain undecayed pending
  the separate re-wear reset work.
- Added opt-in per-person sequential CUSUM detection with hysteresis.
- Ran the 2K → 10K → 25K population ladder against the recalibrated aggregation
  layer, and recorded it in `docs/OPERATIONAL_DETECTION.md`. The outbreak
  becomes detectable between the 2K and 10K rungs (warranted detections 2 → 55 →
  221, disease true positives 0 → 8 → 40, toxin time-to-detection 131 → 81 → 75
  steps, discrimination ~1.0 at both detecting rungs) with no change to the
  sensing layer: `mean_effective_width` is 4.79 at every rung. Broadcasts per
  warranted detection stay flat to improving (10.5 / 10.6 / 8.8), so response
  epsilon tracks detections rather than population, while
  `unexplained_detection_rate` rises from 0.0 to 0.26 at the rungs that detect
  anything.
- Added `examples/detection_power_density_sweep.yaml`, which sweeps
  `wearable_fraction` at the town's fixed 2K population. Each ladder rung raises
  residents and wearables per zone together on a fixed grid, so the ladder alone
  cannot say which of the two the detection gain belongs to. Run against the
  ladder it says that *for broadcasts and plume detections* the zone layer
  responds to the absolute number of wearers rather than to population or to
  adoption share: 2K at 0.6 (1,200 wearers, 45 warranted) and 8K at 0.15 (1,200
  wearers, 46 warranted) land within 2% of each other. Outbreak evidence still
  scales with population in the number of cases available — the outbreak seeds 20
  people at every scale — and across 0.15 to 0.6 the 2K disease arm never exceeds
  one detection.
- Added `examples/detection_power_universal_sweep.yaml`, a near-universal
  adoption counterfactual holding the town's 2K population and sweeping
  `wearable_fraction` 0.85 / 0.90 / 0.95 with subsystem adoption unchanged. It
  narrows the previous density finding: 1,200 wearers was the outbreak's
  detection floor rather than a 2K population ceiling, and at 1,700+ wearers the
  2K town detects the outbreak (disease true positives 8 / 2 / 14, disease
  time-to-detection 120 / 214 / 96 steps against 254 at 0.6 and undefined at
  0.15), with 0.85-arm seed replicates at 8–9 detections and 85–120 steps. Plume
  latency is already saturated (toxin 77–82 steps, the 25K rung's 75), cost per
  detection is unchanged (9.4–11.4 broadcasts per warranted detection,
  `epsilon_per_agent_per_day` bounded at 0.062–0.073), and `mean_effective_width`
  stays 4.79, so what remains binding is channels per person rather than observed
  people. Detection counts scatter across seeds within the plateau, but the 0.90
  arm's ~215-step disease latency replicates across two seeds against 85–120 for
  0.85, so latency is not smooth in adoption over this range and is unexplained.
  It is a simulation capability ceiling, not an adoption forecast.
- Calibrated the cold-start covariance prior independently for each core
  biometric channel from mature benign residual variance in GARLAND's own
  physiology model. The calibration fixes the former shared-prior
  fever-blindness defect; it is a simulation-testbed calibration, not a claim
  about real wearable variance or a formal privacy/security property. Added
  `scripts/coldstart_variance_check.py` to make future physiology-model drift
  visible.
- Added advancing simulated month calculation using the existing 365-day
  convention, plus optional device-local baseline maturation. Fleet-start
  devices can learn configurable prior history with uniform or per-device
  history lengths and a dedicated RNG stream; maturation is evaluation-only,
  has no protocol visibility or privacy-budget cost, and does not create
  detection events. Evaluation summaries report configured cadence and bounded
  maturity coverage. The default remains disabled with zero history.
- Recalibrated the aggregation layer against the corrected token rate. Zone
  broadcasts can now be withheld until the alarm scales freeze
  (`alarm_calibration.defer_broadcasts_until_frozen`,
  `--defer-broadcasts-until-calibrated`; off by default and enabled in
  `examples/detection_power_town.yaml`), because 9,011 of the 2K town's
  9,873 zone triggers were minted before the freeze from cuts the run was still
  establishing were miscalibrated, and each spent response epsilon. Gated, the
  same scenario issues 899 broadcasts instead of 9,816 while keeping 20 of 27
  warranted detections. `privacy.threshold_m` in
  `examples/detection_power_town.yaml` moves 5 → 8, the knee of a
  detections-versus-volume sweep (26/2562 at 3, 20/899 at 5, 17/382 at 8, 4/201
  at 12). `time_window_steps`, dilation and the aggregate-count evidence floor
  were left alone: dilation suppression moved only 0.069–0.112 across the sweep.
  Device density, not the trigger count, is what binds — at the town's authored
  `wearable_fraction` of 0.15 the layer finds one warranted detection at every
  count from 3 to 20. The gate is opt-in because it is only sound when the run
  reaches the freeze and its hazards arrive after it: a run shorter than the
  calibration window never broadcasts at all under it.
- Added opt-in fleet-level zone trigger-count calibration
  (`zone_threshold_calibration`, `--zone-threshold-calibration`), which
  re-derives the count from the quiet-window token counts the aggregator sees
  against a target false-trigger rate and freezes it, instead of using the
  configured `privacy.threshold_m`. It learns one pooled count for the whole
  fleet, and a hazard inside its window inflates what it learns.
- Added a physiology-calibrated toxin exposure truth gate. The default 2.0 bpm
  respiratory-delta threshold corresponds to concentration `c > 0.1`; the
  perturbation remains continuous below that evaluation-only gate, while a
  same-curve 0.1 bpm negligibility floor prevents sub-perceptual doses from
  moving sensors or claiming `TOXIN` provenance. The legacy `0.01` mode applies
  to both floors.
  An explicit `legacy_0_01` mode reproduces pre-calibration results but is not
  recommended. Detection events now expose evaluation-only dosed-agent counts
  and a counter for toxin true positives with fewer than two dosed devices.
  Added a maintained plume-footprint calibration harness and scaled placement
  margins for small grids.
- Recalibrated the staged plume examples to release rate 200, stability D, and
  random-walk mobility. The calibrated footprint is 2.66 ha, about 725 m
  downwind by 40 m crosswind, with approximately 10 above-gate wearables per
  active step at committed density. CI staged-hazard guards now use a
  density-preserving 2,000-agent, 1.74 km-grid downscale with 60% wearable
  adoption rather than a sparse 100-agent, 2 km-grid scenario.
- Added fleet-level alarm-rate calibration (`alarm_calibration`,
  `--no-alarm-calibration`) so the quiet-epoch alarm rate stays flat in
  observation width. The degrees-of-freedom cut assumed jointly Gaussian
  residuals; on a hazard-free run with every subsystem adopted the cut that
  should flag 1.56% of quiet epochs flagged 4.5% at 6–12 channels and 9.4% at
  21–30, so wider vectors looked more sensitive largely because they alarmed
  more often on nothing. The fleet now measures the distance-to-cut ratio over a
  quiet window, reads off the quantile matching the target rate per width
  bucket, and freezes that scale: post-freeze rates were 0.0102–0.0175 against a
  0.0156 target. The correction is floored at 1.0, capped by `max_scale`, and
  reported under `detection_power.alarm_calibration`. It costs sensitivity —
  scoring identical town epochs both ways, the 6–12 bucket moved from FPR 0.0265
  / TPR 0.083 to FPR 0.0066 / TPR 0.042, and zone-level warranted detections fell
  because `privacy.threshold_m` was chosen against the uncorrected token volume.
- Added an aggregate noisy-count content round as the default response
  mechanism. Devices send truthful matches to the trusted central aggregator;
  privacy protection applies to one sensitivity-one Laplace count released per
  broadcast, not to an individual reply against that aggregator. Releases are
  clamped and rounded (with documented upward bias near zero), detection uses a
  configurable one-sided evidence threshold, and aggregate epsilon is charged
  once per release. Randomized response remains selectable for historical
  comparisons.
- Corrected indicative composition accounting to use the tighter of basic and
  advanced composition, so a single release is charged exactly its configured
  epsilon. This also changes small-query RR totals; it is an accounting
  correction, not a mechanism regression, and does not prove privacy for
  data-triggered broadcasts.
- Added an explicit aggregate evidence floor and protocol-visible release
  outcomes for `no_cluster` versus `cluster_below_floor`. The toxin-only
  staged scenario now records the measured tradeoff: aggregate mode produced
  243 releases with a median true cluster size of 1, 102 zero-anomaly releases,
  and only 26 clusters of at least four; historical RR produced two toxin
  true-positive events affecting 1 and 4 agents, with total epsilon 832.8
  versus 243 aggregate release epsilon. The earlier RR `time_to_detection`
  result at step 238 rested on a single-device confirmation that the aggregate
  floor intentionally does not treat as evidence.
- Added detection-power instrumentation stratified by what each person was
  actually wearing when they were scored: `metrics.summary()["detection_power"]`
  now reports per-epoch true- and false-positive rates, mean effective width and
  first-token latency in four effective-width buckets (1–5, 6–12, 13–24, 25+),
  plus each subsystem's reporting yield (`observed_channel_fraction`,
  `masked_channel_fraction`, `reporting_epoch_fraction`) and outcome rates among
  its owners. Effective width counts only channels that were both present and
  unmasked for the epoch, so structural missingness and duty-cycle masking both
  move a person between buckets over a day. The system-level episode metrics were
  unable to say whether adopting a subsystem bought any detection power; these
  can.
- Added an optional drop-one-channel ablation
  (`detection_power.channel_ablation_rate`, `--channel-ablation-rate`) that
  re-scores a sample of alarming epochs with each observed channel removed, at
  the width-corrected cut for the reduced vector, and reports per-channel alarm
  retention and marginal contribution. It answers whether detection is genuinely
  collective: a channel whose removal cancels most of the alarms it appeared in
  would mean the fleet is a single-channel detector wearing a costume. The probe
  draws from its own generator and runs against the pre-update baseline, so
  enabling it changes neither the random stream nor any token. Only the instant
  detector is probed, since a single-epoch re-score cannot say what a
  path-dependent CUSUM would have done.
- Added `examples/detection_power_town.yaml` (2K mixed-modality community),
  `examples/detection_power_ladder_sweep.yaml` (2K → 10K → 25K population ladder) and
  `examples/detection_power_adoption_sweep.yaml` (core-only vs one band vs whole fleet
  at fixed population). Sweep tables now carry `dp_*` columns for mean effective
  width and per-bucket TPR/FPR/latency.
- Added a shared `hypovolemia` signature axis and a `heat_strain_axes`
  constructor, and wired the heat-wave confounder into the band channels. Volume
  depletion is the first state with no device of its own: febrile insensible
  loss, diarrhoeal loss, exertional sweat loss and heat strain all converge on
  it, and it then lengthens `pep_ms` (+15 ms) while lowering `pwv_m_s`,
  `eit_perfusion_pulsatility_ratio` and `bladder_filling_impedance_shift`. Two of
  those oppose the inflammatory drive, so a dehydrated fever understates its own
  severity rather than amplifying it, and a heat wave — which looks fever-shaped
  to the core vitals — now disagrees with an infection on the vascular channels.
  `eit_perfusion_pulsatility_ratio` and `bladder_filling_impedance_shift` stop
  being inert as a result. No dehydration channel or device was added; the
  magnitudes are deliberately sub-illness-scale.
- Added a `headband_eeg` device with five scalar sleep and vigilance channels
  (`sleep_onset_latency_min`, `waso_minutes`, `rem_sleep_fraction`,
  `slow_wave_activity_fraction`, `alpha_theta_ratio`), a `neural` channel system,
  and the `rem_suppression`, `slow_wave_drive` and `cortical_slowing` signature
  axes. The four staging aggregates are event-gated together at wake, since they
  are one scoring pass over one night rather than per-epoch samples, and the
  waking spectral ratio is the most motion-fragile channel in the fleet.
- Made `slow_wave_activity_fraction` the headband's only cause-discriminating
  channel: infection intensifies slow-wave sleep (+7 points) while a benign
  wrecked night suppresses it (−10 points), whereas onset latency, WASO and REM
  loss move the same way under both. `sleep_disturbance` now also drives onset
  latency and WASO, so the actigraph and the headband share one fragmentation
  state instead of counting a bad night twice.
- Split the lumped `adventitious_breath_fraction` acoustic channel into
  `wheeze_duration_fraction` (conducting-airway obstruction) and
  `crackle_count_per_cycle` (parenchymal consolidation), which is the
  discrimination the lumped channel discarded: an irritant plume now wheezes with
  the crackle count at exactly zero, while a pneumonia cracks roughly twice as
  many excursion-cuts as it wheezes. Garment shear fakes crackles only, never a
  tonal wheeze.
- Added `s3_energy_fraction`, `acoustic_motility_index`,
  `eit_perfusion_pulsatility_ratio` and `bladder_filling_impedance_shift`
  channels, with the new `volume_overload`, `airway_obstruction`,
  `parenchymal_consolidation`, `cardiac_contractility`,
  `pulmonary_perfusion_deficit` and `urinary_retention` signature axes and the
  `cardiac_decompensation_axes`, `perfusion_deficit_axes` and
  `urinary_retention_axes` constructors. The thoracic band, abdominal band and
  respiratory patch own them; no hazard or confounder drives the last three axes
  yet, so those channels sit at their resting distributions in current runs.
- Recalibrated `heart_sound_s1_s2_ratio` to published resting statistics
  (1.15 ± 0.22, within-person 0.08) and made it bidirectional: febrile inotropy
  raises it +0.35 while impaired contractility suppresses S1 and drops it −0.55.
  The channel's sign, not its magnitude, now separates decompensation from
  infection.
- Added documentation-only characterization of the five town archetypes and
  the `scripts/characterize_archetypes.py` measurement harness. The report
  records three findings: dilation is computed over residents while only
  wearables respond, the exurb is not the dilution-limited regime (the mill
  town is), and tourist cold-baseline inflow is currently inert. No simulation
  behaviour or defaults changed.
- Added five town-archetype example configurations and a college-town venue
  calibration preset for sweeping detector regimes without changing defaults.
- Added a sensor channel registry (`garland.channels`) and made the observation
  pipeline variable-width and name-addressed: profiles, synthesis, baselines,
  sequential detectors, hazard and confounder deltas, anomaly classification,
  and Open Wearables export all address entries by channel name and carry a
  `ChannelSet`. The default `CORE_VITALS` set reproduces the previous
  four-channel behaviour exactly, including RNG draw order, so seeded runs are
  unchanged.
- Added degrees-of-freedom calibration for anomaly thresholds
  (`garland.thresholds`). A configured Mahalanobis cut is now interpreted as a
  per-epoch false-positive *rate* at the four core vitals and re-expressed at
  whatever width is actually scored, so widening the observation vector no
  longer inflates the alarm rate; the CUSUM slack is rescaled the same way,
  since the resting mean distance grows like `sqrt(dof)`. The default
  four-channel cut of 3.5 and slack of 2.0 are returned unchanged.
- Added missing-channel handling to `BaselineTracker` and
  `CitizenAgent.observe_and_detect` via an `observed` boolean mask. Channels a
  device did not report are marginalized out of the Mahalanobis score rather
  than imputed, leave their baseline, cyclical profiles and covariance entries
  untouched, and shrink no other channel's variance (covariance entries are
  weighted by per-pair observation counts). An all-missing epoch reports
  nothing. Full-mask behaviour is identical to the previous unmasked path.
  Anomaly classification is masked the same way: a channel the device did not
  report neither counts as an excursion nor satisfies a rule arm that requires a
  channel to have stayed quiet. Cross-covariances are clipped to the bound the
  variances imply, since per-pair normalization could otherwise leave the scored
  sub-matrix indefinite (and the distance NaN) for a rarely reported channel.
- Added device-level sensor modalities (`garland.devices`): a person adopts
  *devices*, each of which reports a bundle of derived channels with its own
  usable duty cycle. Enabling `devices.adoption` (config) or `--device-adoption
  KIND=FRACTION` (CLI) widens the fleet's single observation layout to the union
  of the adopted kinds' channels and drives the per-epoch `observed` mask, so a
  non-owner's channels are permanently missing and an owner's can drop out on
  motion, sleep/wake state, or event windows. Catalogued kinds are the historical
  `wrist_ppg`, a `thoracic_eit_acoustic_band`
  (`regional_ventilation_heterogeneity`, `pep_ms`, `pwv_m_s`) and an
  `abdominal_acoustic_band` (`bowel_sound_burst_rate`,
  `gastric_emptying_index`, reported only at postprandial completion). Resting
  distributions, epoch noise and yields follow the calibration table in
  `docs/SENSOR_MODALITIES.md`. The default configuration keeps the four core
  vitals unchanged.
- Added independent per-subsystem batteries and lifecycle state
  (`garland.device_lifecycle.SubsystemLifecycle`): every adopted device kind runs
  its own lifecycle engine, so a flat, removed, or powered-off band masks only
  its own channels while the watch on the same wrist keeps reporting, and
  charging recovers only the subsystem that ran down. Each kind carries a
  `SubsystemPowerProfile` that scales the one `device_lifecycle` block by its own
  draw, capacity, activity sensitivity, and removal habits (the thoracic EIT band
  drains fastest, the watch slowest). The wrist device keeps its historical
  per-person `CitizenAgent.device_status` path and existing `wearables_*`
  metrics; per-subsystem counts and mean battery are reported as
  `subsystem_<kind>_*`. Reporting eligibility now follows the observed-channel
  mask rather than wrist status alone, so a band owner whose watch is dead still
  observes and learns. Runs without `devices.enabled` are unchanged.
- Added illness and confounder signatures for the EIT/contact-acoustic channels
  (`garland.modality_signatures`), so adopting a band now adds detection power
  rather than only width. Causes are expressed as four latent axes
  (inflammatory drive, pulmonary involvement, enteric drive, arterial
  stiffening) and only then mapped to whichever calibrated channels the running
  set contains: symptomatic infection shortens `pep_ms`, delays
  `gastric_emptying_index`, raises `pwv_m_s` and
  `regional_ventilation_heterogeneity`, while a pathogen's new
  `SEIRConfig.enteric_involvement` (0.9 for norovirus) moves that tropism from
  ventilation to `bowel_sound_burst_rate`. One shared arterial axis keeps a
  fever from being counted twice across vascular channels. Background ILI and
  exercise carry the same axes at lower severity so the bands are not a free
  benign-versus-outbreak discriminator, while irritant exposure raises
  ventilation heterogeneity without fever or gastric delay and contact artifact
  perturbs the impedance field alone. Core-vitals runs are unchanged.
- Added a `motion_actigraphy` device kind reporting `step_count` (per epoch) and
  `sleep_fragmentation_index` (one nightly aggregate, event-gated at wake), with
  its own battery profile so its depletion masks only the behavioural channels.
  Two new latent axes carry it: `activity_withdrawal` and `sleep_disturbance`.
  These are deliberately asymmetric — sickness behaviour removes about 2,900
  steps/day while a single exercise bout adds hundreds of steps to one epoch — so
  the pedometer is detectable through a sustained shortfall rather than any one
  epoch, and a benign disrupted night fragments sleep and slows the next day.
  Both axes move earlier in the course than the febrile ones, modelling prodromal
  malaise; irritant exposure and contact artifact leave them untouched, keeping
  the toxin-versus-disease separator intact. Perturbed observations are now
  clamped to per-channel physical floors, since a pedometer cannot report a
  negative count. Core-vitals runs are unchanged.
- Added an `instrumented_footwear` device kind reporting `gait_speed_m_s`,
  `stride_time_variability` and `gait_asymmetry`, with its own battery profile.
  It is the first *ambulation-gated* modality: `DeviceChannel.activity_bonus`
  inverts the artifact-driven yield model, so a shoe reports ~10% of sedentary
  epochs, 65–90% mid-bout, and nothing overnight — motion is the precondition
  rather than the artifact. Gait speed reads the existing
  `activity_withdrawal` axis (malaise −0.15 m/s, an exercise bout +0.45 m/s) so
  reduced ambulation is not counted twice, and a new `neuromotor_fatigue` axis
  raises stride-time variability. The two channels therefore agree in illness
  and disagree under exertion, which is what separates them. `gait_asymmetry`
  carries no illness signature at all: it is driven only by a new
  `instrument_artifact` axis, giving the fleet a negative-control channel where
  an alarm means the footwear changed rather than the wearer. Core-vitals runs
  are unchanged.
- Added a `respiratory_acoustic_patch` device kind reporting `cough_rate`,
  `speech_pause_ratio`, `adventitious_breath_fraction` and
  `heart_sound_s1_s2_ratio` as derived scalars — no waveform, spectrogram or
  speech content exists anywhere in the model. Yield spans the widest range of
  any device so far, because a cough is loud enough to survive motion that
  buries a heart sound: ~88% of epochs for cough against ~5% for heart sounds
  mid-activity, with speech gated to waking epochs and breath sounds best
  overnight. A new `airway_irritation` axis drives cough, kept separate from
  `pulmonary_involvement` because a consolidated lobe can be quiet while an
  inhaled irritant coughs violently without consolidating anything — and an
  irritant plume moves cough *harder* than an infection does, so cough is
  deliberately not a toxin-versus-disease discriminator (the absent
  `inflammatory_drive` still is). Speech fragmentation and adventitious breath
  sounds read the existing `pulmonary_involvement` axis and heart sounds the
  existing `inflammatory_drive`, so neither consolidation nor fever is counted
  twice. `contact_artifact_axes` now runs entirely through
  `instrument_artifact`, which drives ventilation heterogeneity, gait asymmetry
  and a false crackle fraction at unchanged magnitudes: an artifact can reach
  the transducers it corrupts, and can no longer masquerade as pulmonary
  physiology on channels it never touches. Core-vitals runs are unchanged.
- Added a `chest_electrode_patch` device kind reporting `qtc_ms`,
  `ectopy_burden` and `ptt_systolic_bp` — two-lead ECG interval and beat
  statistics plus a cuffless systolic estimate from electrode-to-pulse-foot
  transit time, all as derived per-epoch scalars with no ECG waveform stored
  anywhere. It is the first device to *re-report* channels another device
  already covers: `heart_rate` and `hrv_rmssd` come from the electrodes at
  higher yield than the wrist manages, and because observation masks are OR-ed
  across owned devices, an owner whose watch battery has flattened keeps
  reporting rate and variability. `ptt_systolic_bp` reads the existing
  `arterial_stiffening` axis, so it and `pwv_m_s` move together instead of
  counting one vascular shift twice — including downward together in
  distributive shock. `qtc_ms` reads `inflammatory_drive` plus upward
  `enteric_drive`, which is what lets a chest patch see a gastroenteritis at
  all: electrolyte loss prolongs the QT interval while the respiratory channels
  quieten. `ectopy_burden` is deliberately weak in both directions — lead noise
  and motion transients (`instrument_artifact`) fake premature beats about as
  hard as inflammation produces them, so it is informative only in company, and
  the QT and pressure channels are what disambiguate it. Yield is gated on
  motion in the order the physics implies (R-peaks survive, T-waves do not,
  pulse feet least of all), and the subsystem has its own battery profile.
  Core-vitals runs are unchanged.
- Added disabled-by-default block-fire smoke and stadium/civic-victory
  confounder generators with evaluation-only footprints and cause-labelled
  warrant classifications.
- Added evaluation-only exposure attributes, structured diurnal heat
  advisories, jittered sleep-disruption onset, additive warrant-class metrics,
  and any-match benign ask scoring. Exposure attributes remain outside all
  protocol objects, trigger logic, classification, and ask vocabulary; see
  `docs/EVENT_CATALOGUE.md`.
- Calibrated the default heat-wave night floor to `0.35` so overnight
  material exposure remains observable for vulnerable, non-air-conditioned
  agents without broadening the evaluation footprint to merely uncooled
  occupants.
- Disabled-by-default benign confounder engine with independent
  exercise/sleep-disruption/sensor-artifact sources, an optional all-zone
  heat-wave instance, seeded RNG isolation, cause-labelled metrics, and
  configuration/CLI support. This phase is label-only and does not alter
  hazard classification or scoring.
- Added opt-in venue-crowding and exogenous household background-ILI
  instances, plus model-side benign overlap and attribution metrics. These
  metrics remain outside protocol payloads and hazard classification.
- Deterministic ordering for background metric group folds and perturbation
  cause aggregation, making seeded summaries reproducible across processes
  without changing metric values.
- Optional second-round disambiguation queries with open-ended hypotheses,
  content-free aggregate acknowledgements, human-approved yes/no answers,
  expiry/unresolved accounting, and separate privacy metrics.
- Reworked disambiguation triggering to use only protocol-visible cluster shape
  for `RECENT_ADOPTION` and `AMBIENT_HEAT`, and added model-side reporting of
  well-founded, unfounded, and unscored asks. Well-founded plus unfounded plus
  unscored equals issued queries; unfounded and unscored ask epsilon are
  reported separately.
  The previous onboarding-age gate was oracle-validated, invalidating the
  previously published disambiguation numbers; removing
  `min_onboarding_wearables_in_zone` is a breaking change to a default-off
  experimental feature. Unfounded asks are reported but not penalized in the
  discrimination score pending evaluation of realistic unfounded-ask rates,
  epsilon expenditure, and whether such asks should eventually carry a cost.
- Realistic mixed-benign evaluation scenario
  `examples/disambiguation_evaluation.yaml` and the operator-run
  `scripts/disambiguation_ask_eval.py` (excluded from pytest and CI), plus the
  measured four-variant ask-quality results in `docs/OPERATIONAL_DETECTION.md`.
  The follow-up query fires on roughly half of all broadcasts and consumes
  roughly half of the run's total epsilon; `recent_adoption` reaches 80.6%
  precision over scorable asks while `ambient_heat` issues 95% of all asks at
  8.0%. Unfounded asks remain reporting-only and outside
  `discrimination_score`.
- Reworked the `AMBIENT_HEAT` trigger from a single-step absolute breadth test
  into breadth sustained over `min_breadth_windows` bins and exceeding
  `breadth_ratio` times a channel breadth baseline (`breadth_baseline_alpha`)
  that excludes world-settling bins, because the absolute floor had been
  calibrated against the un-settled startup period rather than an ambient
  cause. Added an `ask_epsilon_budget` for the follow-up channel, checked
  against epsilon already spent so at most one in-flight ask may overshoot,
  with `disambiguation_asks_suppressed_by_budget` and
  `disambiguation_max_ask_epsilon_delta` published; suppressed asks are never
  issued, answered, or scored. Added overall and per-hypothesis ask precision
  (`well_founded / (well_founded + unfounded)`, unscored excluded) with
  hash-seed-independent key ordering. `RECENT_ADOPTION` is unchanged. On the
  re-measured scenario asks fall from 0.57 to 0.062 per broadcast and the
  channel's epsilon share from 51.2% to 17.6%, `ambient_heat` from 1,072 asks
  at 8.0% precision to 66 asks with no unfounded ask, and a tight-budget
  variant is included in `scripts/disambiguation_ask_eval.py`. Unfounded asks
  remain reporting-only.
- Configurable first-time device adoption schedules for startup, rollout,
  trickle, and household/venue cohorts, with adoption step/zone events,
  not-adopted per-step counts, and onboarding provenance labels. Schedules
  can retain an established initial population and use an explicit
  one-simulated-day onboarding window distinct from the covariance-prior
  regime. Added separate zone-local onboarding-window and
  onboarding-plus-covariance-prior peak metrics.
- Renamed the shipped burn-in markers to explicit world-settling exclusion,
  fleet-cold-start, and device-onboarding labels in summaries, per-step CSV,
  and sweep results; the deprecated `background_burn_in_steps` config key
  remains accepted as an alias
- Corrected device re-adoption so retained baselines are not charged a new
  local warm-up by default; the legacy reset remains available through
  `warmup_on_device_adopt`
- Burn-in-aware background assessment with full-run and settled rate,
  dispersion, threshold-tail, and population VMR fields, retaining bounded
  scalar and histogram folding
- Background-only anomaly-rate and heterogeneous-Poisson dispersion metrics,
  including occupancy-stratified variance-to-mean ratios and threshold-tail
  comparisons in summaries and sweep results
- Explicit emission-bin and aggregation-window background statistics, including
  bounded scalar/histogram folding and per-agent background-token summaries
- Seed-42 startup, diurnal, and activity/circadian ablation diagnostics for
  separating settled null behavior from the startup transient
- A reproducible null-baseline background assessment sweep configuration
- Model-side cause labels for biometric perturbations and measurement-only cause
  attribution metrics without changing protocol objects or default behavior
- CI guards for undefined metrics, hazard reachability, privacy-mechanism
  sensitivity, and long-run null-baseline stationarity
- Reproducible null and staged detection scenarios with configurable anomaly thresholds and operator-facing daily detection metrics
- Provenance-only attributed versus coincidental detection metrics, including
  affected-token fragmentation summaries
- Scenario- and seed-scoped operational detection measurement records and
  updated architecture, privacy, review, and testing guidance
- `garland.paths` helpers for validating user-supplied filesystem paths (SonarQube S2083)
- `uv.lock` for reproducible dependency resolution (SonarQube SCA / supply-chain analysis)
- Wearable device lifecycle: battery depletion, user power-off, device removal, and home charging (`--enable-device-lifecycle`, `examples/device_lifecycle.yaml`)
- GitHub Actions CI: pytest with coverage, ruff lint, mypy type checking
- CLI integration tests and simulation protocol integration tests
- `CONTRIBUTING.md` and `CHANGELOG.md`
- Public `SpatialGrid.cell_ids` property

### Changed
- Optimized rectangular and H3 dilation ring traversal without changing zone
  ordering or accounting-visible population metrics.
- Changed respondent-basis scaling guards from machine-specific absolute
  budgets to same-process resident/observed-device runtime ratios, retaining
  generous absolute catastrophe ceilings; local measurements remain reported
  so the operational respondent-basis cost stays visible.
- Made strict under-k broadcast-release enforcement configurable and
  disabled by default. Under-k release counts, positive-reply coverage, and
  epsilon burn remain reported even when enforcement is off.
- Changed the default randomized-response truthfulness from `0.75` to `0.5`
  (`ε=1.099` per response and unaffected-positive probability `0.25`).
  Lower values reduce epsilon but also reduce released signal excess over the
  randomized-response null.
- Changed response-epsilon accounting to use the randomized-response
  mechanism-derived value by default. At `p=0.75`, previously published
  epsilon figures understated that mechanism by approximately 19x. Planar
  Laplace geo epsilon is reported separately as a per-metre parameter rather
  than silently omitted.
- Changed the default k-anonymity dilation basis to protocol-observed
  respondent estimation. Published epsilon and dilated zone footprints
  therefore move: the estimator accepts wider zones and higher epsilon per
  answer to avoid overstating the available respondent population. Triggers
  whose conservative estimate cannot reach `k_min` at the spatial bound are
  now suppressed rather than broadcast citywide.
- Fleet cold-start now means cold-baseline behavior reached the protocol,
  rather than merely that trackers were constructed cold; covariance-prior
  occupancy (`BaselineTracker.n_samples < 5`) is documented separately from
  baseline convergence
- Device-adoption warm-up CLI flags are mutually exclusive, and lifecycle
  re-adoption metrics now have a single recorder source of truth
- README privacy section reframed as design goals with simulation disclaimer
- README attack section documents all five attack types including replay
- Removed unused runtime dependencies (`neurokit2`, `scipy`, `h3`, `pydantic`)
- License aligned to Apache 2.0 across README, `pyproject.toml`, and `LICENSE`

### Fixed
- SonarQube S1244: disease perturbation gate uses ``np.isclose`` instead of
  float ``!=`` in ``GarlandModel._agent_perturbation_contributions``
- SonarQube S8707: path helpers use an analyzer-visible ``realpath`` +
  ``startswith(base + os.sep)`` sanitizer (absolute CLI paths remain allowed
  after ``realpath``) so I/O sinks no longer hide containment behind a boolean
  helper
- The Poisson tail behind the background assessment now stops summing once the
  remaining mass is negligible, so its cost is set by the rate rather than by
  `privacy.threshold_m`. A large configured trigger count previously turned the
  summary into a multi-billion-iteration loop
- `--channel-ablation-rate 0.0` now switches off an ablation rate set in a config
  file; zero is the meaningful "off" value, so it can no longer double as the
  flag's unset sentinel
- `garland sweep` now reports the directory it actually wrote
  `sweep_results.csv` to, rather than always naming the default `output/sweep`
  even when the sweep config set `output_dir`
- Background assessment now uses shared simulation-day timing for daily buckets
  and its default world-settling exclusion, with settled metrics covered end
  to end
- Stationary baseline residual centering/covariance estimation and reachable toxin respiratory classification
- Sequential detector episodes now clear below a configurable re-arm level and
  continue token emission while active, allowing later independent alarms
- SonarQube S8707: route user-supplied path I/O through validated `garland.paths` helpers
- SonarQube S1244: use `pytest.approx` for floating-point assertions in tests
- SonarQube code smells: reduce cognitive complexity, remove unused variables, deduplicate literals
- Added `networkx` runtime dependency (required by Mesa)
- Zone ID namespace mismatch in privacy protocol (grid cell IDs throughout)
- Episode-granular FN/TN metrics (no per-step inflation)
- Zone-local plume classification for toxin detection
- Sybil and deanon attack metrics wired into summary output
- CARDIAC anomaly detection classification in metrics
- Household/neighborhood spatial model alignment
- `plot_metrics` docstring matches actual plot outputs

### Removed
- Dead `MaliciousAgent` class (attacks live in `attacks.py`)

## [0.1.0] — 2024

Initial release of the GARLAND epidemiological security testbed:

- Mesa-based agent simulation at 250K agent scale
- SEIR infectious disease and Gaussian plume hazard models
- Decentralized privacy protocol (tokens, K-anonymity dilution, broadcast queries)
- Attack simulation layer (Sybil, deanonymization, correlation, eclipse)
- CLI entry point with CSV/JSON/plot outputs
- Pytest suite for privacy primitives and simulation smoke tests

[Unreleased]: https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/compare/v0.1.0...main
[0.1.0]: https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/releases/tag/v0.1.0
