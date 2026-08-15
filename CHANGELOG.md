- Added opt-in per-person sequential CUSUM detection with hysteresis.
# Changelog

All notable changes to GARLAND are documented here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
- SonarQube S1244: disease perturbation gate uses ``np.isclose`` instead of
  float ``!=`` in ``GarlandModel._agent_perturbation_contributions``
- SonarQube S8707: path helpers use an analyzer-visible ``realpath`` +
  ``startswith(base + os.sep)`` sanitizer (absolute CLI paths remain allowed
  after ``realpath``) so I/O sinks no longer hide containment behind a boolean
  helper
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
