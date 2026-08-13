- Added opt-in per-person sequential CUSUM detection with hysteresis.
# Changelog

## Unreleased

- Corrected confounder attribution to keep detection routing independent of
  model-side provenance, and added multi-day ILI calibration and cooking funnel
  measurements.
- Calibrated the cooking example with schedule-driven home anchors and added
  event-level household reach to the exposure funnel.

All notable changes to GARLAND are documented here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Opt-in clustered background ILI and household dinner-time cooking-irritant
  confounders with cause-labelled biometric perturbations and burden metrics
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
- README privacy section reframed as design goals with simulation disclaimer
- README attack section documents all five attack types including replay
- Removed unused runtime dependencies (`neurokit2`, `scipy`, `h3`, `pydantic`)
- License aligned to Apache 2.0 across README, `pyproject.toml`, and `LICENSE`

### Fixed
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
