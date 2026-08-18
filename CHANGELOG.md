- Added opt-in per-person sequential CUSUM detection with hysteresis.
# Changelog

All notable changes to GARLAND are documented here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
