# GARLAND Known Issues Catalog

Issues filed from the initial code review. Use this when triaging or implementing fixes.

## #2 — Detection classification uses global plume timing

**Label:** bug  
**Files:** `src/garland/simulation.py` (`_classify_detection`)

`_classify_detection()` marks respiratory detections as toxin TPs when the plume is globally active by timestep, not when agents in the query zone have `concentration > threshold`.

**Fix direction:** Classify using zone-local plume exposure from `compute_plume_concentration()`.

---

## #3 — Missing networkx dependency

**Label:** bug  
**Files:** `pyproject.toml`

Mesa imports `networkx`; fresh `pip install -e ".[dev]"` fails test collection.

**Fix direction:** Add `networkx` to runtime dependencies.

---

## #4 — Attack layer not integrated

**Label:** bug  
**Files:** `src/garland/simulation.py`, `src/garland/attacks.py`, `src/garland/app.py`

Only Sybil runs in `GarlandModel.step()` by default. Deanonymization runs when enabled via config.

**Fix direction:** Wire attacks into step loop or remove from CLI/README.

---

## #5 — Zone ID namespace mismatch (P0)

**Label:** bug  
**Files:** `src/garland/agents.py`, `src/garland/simulation.py`, `src/garland/spatial.py`

Tokens use `neighborhood_id` as `zone_id`; dilution uses grid `cell_id`. Query matching compares neighborhood IDs to dilated cell ID lists.

**Fix direction:** Use `grid.cell_of(agent.idx)` consistently for tokens, dilution, and query matching.

---

## #6 — License inconsistency

**Label:** documentation  
**Files:** `README.md`, `pyproject.toml`, `LICENSE`

README and pyproject say MIT; LICENSE file is Apache 2.0.

**Fix direction:** Align all three to one license.

---

## #7 — Dead metrics fields

**Label:** bug  
**Files:** `src/garland/metrics.py`, `src/garland/simulation.py`

`total_queries_issued`, `sybil_false_alerts`, `deanon_attempts`, `deanon_successes` never updated; `summary()` reports zeros.

**Fix direction:** Wire updates in simulation step or remove from summary.

---

## #8 — FNR inflated by per-step counting

**Label:** bug  
**Files:** `src/garland/simulation.py`, `src/garland/metrics.py`

`record_missed_detection()` called every step hazard is active without detection, multiplying FN counts.

**Fix direction:** Episode-level or first-miss-only FN tracking; document metric definition.

---

## #9 — Household/wearable spatial model

**Label:** documentation  
**Files:** `src/garland/simulation.py`

`household_ids = np.arange(n) // household_size_mean` ignores neighborhood clustering.

**Fix direction:** Assign households within neighborhoods, or update README.

---

## #10 — Unused dependencies

**Label:** enhancement  
**Files:** `pyproject.toml`, `README.md`

`neurokit2`, `scipy`, `h3`, `pydantic` declared but not imported. README claims NeuroKit2/H3 integrations.

**Fix direction:** Remove deps and update README, or implement integrations.

---

## #11 — No performance validation at 250K

**Label:** enhancement  
**Files:** `src/garland/simulation.py`, docs

No benchmark for default CLI scale. Position init uses Python loop; SEIR samples max 500 infectious agents/step.

**Fix direction:** Optional slow benchmark; vectorize init; document runtime expectations.

---

## #12 — Test coverage gaps

**Label:** enhancement  
**Files:** `tests/`

No CLI tests, no end-to-end protocol test, weak assertions in privacy tests.

**Fix direction:** Integration test for token → broadcast → response; strengthen bounds.
