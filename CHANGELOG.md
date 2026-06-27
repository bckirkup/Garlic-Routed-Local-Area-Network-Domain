# Changelog

All notable changes to GARLAND are documented here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
- SonarQube S6729/S1244: use `np.nonzero` instead of single-arg `np.where`; avoid direct float equality on NumPy arrays
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
