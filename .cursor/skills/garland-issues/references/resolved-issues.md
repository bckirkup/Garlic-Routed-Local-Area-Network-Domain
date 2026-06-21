# GARLAND Resolved Issues — Regression Checklist

Use this when reviewing PRs or running `/code-review`. **Do not re-introduce these bugs.**

Verify current open issues: `gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open`

---

## Wave 1 — Initial hardening (closed #2–#16)

| Issue | Type | Fix summary | Regression test |
|-------|------|-------------|-----------------|
| #2 | Bug | Toxin TP uses `_zone_has_plume_exposure()`, not global plume timestep | `TestDetectionClassification` |
| #3 | Bug | `networkx` in `pyproject.toml` deps | Import/collection succeeds |
| #4 | Bug | All 5 attacks wired in `GarlandModel.step()` + CLI | `TestAttackSummaryMetrics` |
| #5 | Bug | Tokens/dilution/queries use grid **cell IDs** | Protocol integration tests |
| #6 | Doc | License aligned to Apache 2.0 | — |
| #7 | Bug | Attack metrics synced to `MetricsCollector.summary()` | Attack summary tests |
| #8 | Bug | Episode-granular FN/TN, not per-step | `test_metrics.py` |
| #9 | Bug | Households nested within neighborhoods | Wearable/household tests |
| #10 | Enhancement | Removed unused deps (neurokit2, scipy, h3, pydantic) | — |
| #11 | Enhancement | Benchmark module + `docs/SCALING.md` | `test_scaling.py` |
| #12 | Enhancement | CLI, metrics, attack, integration tests (110 total) | Full suite |
| #16 | Bug | CARDIAC anomaly classification in metrics | Cardiac detection tests |

---

## Wave 2 — Post-v0.2 review (closed #24–#39)

| Issue | Type | Expected behavior after fix |
|-------|------|----------------------------|
| #24 | Bug | Summary ε matches adaptive composition **or** README describes linear accounting |
| #25 | Bug | FEBRILE/MULTI_SYSTEM TP uses `_zone_has_active_disease(zone_cells)` |
| #26 | Enhancement | Public `SpatialGrid` accessor; no `_cell_ids` reads outside `spatial.py` |
| #27 | Enhancement | `MaliciousAgent` removed from `agents.py` |
| #28 | Doc | `CONTRIBUTING.md` and `CHANGELOG.md` present |
| #29 | Feature | Agent mobility (if implemented) |
| #30 | Feature | Parameter sweep / experiment runner (if implemented) |
| #31 | Enhancement | `ruff check` in CI workflow |
| #32 | Feature | H3 hex spatial indexing option (if implemented) |
| #33 | Feature | NeuroKit2/OpenWearables integration (if implemented) |
| #34 | Feature | Dockerfile / container docs (if implemented) |
| #35 | Doc | README privacy section framed as design goals |
| #36 | Feature | YAML/TOML config file support (if implemented) |
| #37 | Enhancement | mypy/pyright in CI (if implemented) |
| #38 | Feature | Multi-plume / multi-outbreak config (if implemented) |
| #39 | Doc | `plot_metrics` docstring accurate; replay listed in README attacks |

When reviewing, confirm wave-2 bug fixes (#24, #25) with:

```bash
python3 -m pytest tests/test_simulation.py::TestDetectionClassification -v
python3 -m pytest tests/test_privacy.py::TestAdaptiveComposition -v
```

---

## Quick regression commands

```bash
python3 -m pytest tests/ -v
ruff check src tests
python3 -m pytest tests/test_simulation.py::TestProtocolSimulationIntegration -v
python3 -m pytest tests/test_simulation.py::TestAttackSummaryMetrics -v
python3 -m pytest tests/test_metrics.py -v
```

---

## Filing new bugs

If a regression reappears, reopen the original issue or file a new one with:

```markdown
## Type
**Bug fix**

## Regression
Reintroduces fix from #N — describe how.
```
