# GARLAND Issue Catalog

Last updated: after v0.2 hardening (110 tests, ~91% coverage, CI on main).

Use **Type** to decide how to work an item:
- **Bug fix** — incorrect behavior; fix and add regression test
- **Enhancement** — cleanup, refactor, or engineering improvement
- **Documentation** — docs/code mismatch; no logic change required
- **Feature** — new capability; may need design discussion

---

## Open — Bug fixes (priority)

| Issue | Priority | Title |
|-------|----------|-------|
| [#25](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/25) | **P1** | FEBRILE/MULTI_SYSTEM use global disease check, not zone-local |
| [#24](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/24) | **P1** | Privacy budget summary uses linear ε sum, not adaptive composition |

### #25 — FEBRILE/MULTI_SYSTEM global disease classification

**Type:** Bug fix  
**Files:** `src/garland/simulation.py` (`_classify_detection`)

RESPIRATORY and CARDIAC use `_zone_has_plume_exposure()` / `_zone_has_active_disease()`. FEBRILE and MULTI_SYSTEM still check global infectious count.

**Fix:** Use `_zone_has_active_disease(query.zone_cells)` for TP classification.

---

### #24 — Linear epsilon vs adaptive composition

**Type:** Bug fix  
**Files:** `src/garland/agents.py`, `src/garland/metrics.py`, `README.md`

`compute_adaptive_composition_epsilon()` is tested but unused in summary output. Runtime sums `epsilon_per_response × responses` linearly while README claims adaptive composition.

**Fix:** Wire composition into summary **or** update README/plots to match linear accounting.

---

## Open — Enhancements (cleanup / engineering)

_No open enhancement issues — see closed list below._

---

## Open — Documentation

_No open documentation issues — see closed list below._

---

## Open — Features (see also `feature-backlog.md`)

| Issue | Type | Title |
|-------|------|-------|
| [#29](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/29) | Feature | Agent mobility and dynamic cell membership |
| [#32](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/32) | Feature | H3 hexagonal spatial indexing |
| [#36](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/36) | Feature | YAML/TOML config file support |
| [#30](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/30) | Feature | Parameter sweep / experiment runner |
| [#34](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/34) | Feature | Docker reproducible environment |
| [#33](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/33) | Feature | NeuroKit2 / OpenWearables integration |
| [#38](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/38) | Feature | Multi-plume and multi-outbreak scenarios |

Full feature roadmap: `feature-backlog.md`

---

## Closed — Initial review (resolved)

Do not re-introduce these regressions:

| Issue | Topic |
|-------|-------|
| #2 | Zone-local plume detection classification |
| #3 | Missing `networkx` dependency |
| #4 | Attack layer integration (all 5 attack types) |
| #5 | Zone ID namespace mismatch |
| #6 | License alignment (Apache 2.0) |
| #7 | Dead metrics fields in summary |
| #8 | FNR inflated by per-step counting |
| #9 | Household/neighborhood spatial model |
| #10 | Unused dependencies removed |
| #11 | Performance validation / scaling docs |
| #12 | Test coverage gaps |
| #16 | CARDIAC detection classification |
| #26 | Public `SpatialGrid.cell_ids` accessor |
| #27 | Removed unused `MaliciousAgent` class |
| #28 | CONTRIBUTING.md and CHANGELOG.md |
| #31 | Ruff lint in CI |
| #35 | README privacy design goals disclaimer |
| #37 | Mypy type checking in CI |
| #39 | `plot_metrics` docstring and replay attack docs |

---

## Suggested fix order

1. **#25** — zone-local febrile classification (metrics correctness)
2. **#24** — epsilon accounting alignment (privacy reporting)
3. Features per `feature-backlog.md` priority
