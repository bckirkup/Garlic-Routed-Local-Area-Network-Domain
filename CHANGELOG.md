- Added opt-in per-person sequential CUSUM detection with hysteresis.
# Changelog

All notable changes to GARLAND are documented here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
  `docs/SENSOR_MODALITIES.md`; illness signatures for the new channels are not
  wired in yet. The default configuration keeps the four core vitals unchanged.
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
