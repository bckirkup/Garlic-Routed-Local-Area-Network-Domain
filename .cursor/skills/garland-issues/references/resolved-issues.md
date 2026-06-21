# GARLAND Resolved Issues — Regression Checklist

Do **not** re-introduce these fixes. Full history in `CHANGELOG.md`.

Check open work: `gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open`

---

## Wave 1 (#2–#16) — Core hardening

| Issue | Type | Fix | Verify with |
|-------|------|-----|-------------|
| #2 | Bug | Zone-local plume TP (`_zone_has_plume_exposure`) | `TestDetectionClassification` |
| #3 | Bug | `networkx` in deps | `pip install -e ".[dev]"` + pytest collect |
| #4 | Bug | All 5 attacks wired + CLI | `TestAttackSummaryMetrics` |
| #5 | Bug | Grid cell IDs in privacy protocol | Protocol integration tests |
| #6 | Doc | Apache 2.0 license aligned | README/LICENSE/pyproject |
| #7 | Bug | Attack metrics in summary | Attack summary tests |
| #8 | Bug | Episode-granular FN/TN | `test_metrics.py` |
| #9 | Bug | Households in neighborhoods | Household/wearable tests |
| #10 | Enhancement | Trimmed unused deps | pyproject.toml |
| #11 | Enhancement | Benchmark + `docs/SCALING.md` | `test_scaling.py` |
| #12 | Enhancement | Broad test coverage | Full suite |
| #16 | Bug | CARDIAC classification | Cardiac detection tests |

---

## Wave 2 (#24–#39) — v0.2 polish

| Issue | Type | Status on main |
|-------|------|----------------|
| #24 | Bug | Adaptive composition — verify summary/README alignment |
| #25 | Bug | Zone-local FEBRILE via `_zone_outbreak_instance()` |
| #26 | Enhancement | Public spatial accessors (`SpatialIndex`, H3/rect) |
| #27 | Enhancement | `MaliciousAgent` removed |
| #28 | Doc | `CONTRIBUTING.md`, `CHANGELOG.md` |
| #29 | Feature | Agent mobility (`--static-agents` to disable) |
| #30 | Feature | `garland sweep` + `experiment.py` |
| #31 | Enhancement | Ruff in CI |
| #32 | Feature | H3 hex backend (`--spatial-backend h3`) |
| #33 | Feature | Optional NeuroKit2 (`--biometric-synthesis neurokit`, `.[biosignals]`) |
| #34 | Feature | Docker — check if filed / open |
| #35 | Doc | Privacy design goals disclaimer in README |
| #36 | Feature | YAML/TOML via `--config` + `examples/` |
| #37 | Enhancement | Mypy in CI |
| #38 | Feature | Multi-plume / multi-outbreak configs |
| #39 | Doc | Replay in README; plot_metrics docstring |

---

## Regression commands

```bash
ruff check src tests
mypy
python -m pytest tests/ -v
python -m pytest tests/test_simulation.py::TestDetectionClassification -v
python -m pytest tests/test_simulation.py::TestProtocolSimulationIntegration -v
python -m pytest tests/test_mobility.py tests/test_config.py tests/test_experiment.py -v
```

---

## Filing regressions

```markdown
## Type
**Bug fix**

## Regression
Reintroduces fix from #N.
```
