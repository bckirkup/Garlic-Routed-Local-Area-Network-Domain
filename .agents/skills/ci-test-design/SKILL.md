---
name: ci-test-design
description: Design CI tests that verify both golden-value regression (outputs match expected references) and configuration sensitivity (changing a config parameter changes the output). Use when writing or reviewing any test for any repository.
---

# CI Test Design: Golden Values + Configuration Sensitivity

Every CI test should guard against two failure modes:

1. **Golden-value regression** -- an unintended code change silently alters outputs that were previously correct.
2. **Configuration dead-wiring** -- a config parameter is parsed and stored but never reaches the computation, so changing it has no effect on the output.

A test suite that only checks (1) can pass indefinitely even if half the config
keys are ignored. A suite that only checks (2) may not catch a refactor that
breaks previously-validated numerical results. **Good CI tests cover both.**

## Principles

### Golden-value tests (regression anchors)

A golden-value test runs a computation with **fixed inputs** and asserts the
output matches a **known-good reference**. The reference can be:

- A hardcoded expected value (simplest).
- A committed golden file (JSON, HDF5, CSV) compared at CI time.
- A deterministic fingerprint (hash of quantized outputs).

**Rules:**

1. Pin every source of randomness (seed, RNG state).
2. Assert on **quantitative outcomes**, not just crash-free execution.
   Bad: `assert result is not None`. Good: `assert count == 35`.
3. Prefer narrow tolerances. Use `pytest.approx(val, rel=0.01)` or
   `assert abs(x - golden) < eps` rather than wide bounds that hide drift.
4. When a golden value legitimately changes (algorithm improvement, bug fix),
   update the reference **in the same commit** with a clear rationale in the
   commit message.
5. For stochastic simulations, seed-pin and assert on the seed-deterministic
   outcome. If the system is inherently nondeterministic (e.g., MPI race
   conditions), assert on statistical bounds and note why.

**Language-specific patterns:**

C++ (CTest):
```cpp
// Hardcoded golden value
Simulation sim;
sim.init(cfg);
sim.run();
assert(sim.agents().size() == 50);
assert(std::abs(sim.time() - 600.0) < 1e-6);
```

Python (pytest):
```python
# Golden file comparison
history = run_simulation(epochs=24)
assert history[-1]["summary"] == EXPECTED_SUMMARY
assert history[-1]["cost"] == pytest.approx(EXPECTED_COST, rel=0.01)

# Reproducibility (same inputs -> same outputs)
fp_a = run_fingerprint(cfg)
fp_b = run_fingerprint(cfg)
assert fp_a == fp_b
```

### Configuration sensitivity tests (propagation probes)

A sensitivity test verifies that **changing a config parameter changes the
output**. It catches dead wiring: a config key that is parsed but never
plumbed through to the computation.

**Rules:**

1. Start from a shared baseline config. Change **exactly one parameter**.
2. Assert the output **differs** from the baseline.
   For deterministic systems, use fingerprint inequality:
   `assert fp_changed != fp_baseline`.
   For stochastic systems, run N trials and assert statistical difference.
3. For boolean/toggle parameters, test both states and assert different outputs.
4. For continuous parameters, pick a value far enough from the baseline that
   the effect is unambiguous (e.g., 10x the default, not 1.01x).
5. Group related sensitivity tests so adding a new config key has an obvious
   place to add its probe.

**Language-specific patterns:**

C++ (CTest):
```cpp
// Toggle sensitivity
SimulationConfig with_feature = baseline;
with_feature.advection.peristaltic_enabled = true;

SimulationConfig without_feature = baseline;
without_feature.advection.peristaltic_enabled = false;

uint64_t fp_on  = run_fingerprint(with_feature);
uint64_t fp_off = run_fingerprint(without_feature);
assert(fp_on != fp_off);
```

```cpp
// Continuous parameter sensitivity
SimulationConfig tuned = baseline;
tuned.receptor.kill_rate_colicin = 2e-3;  // 10x default

uint64_t fp_tuned    = run_fingerprint(tuned);
uint64_t fp_baseline = run_fingerprint(baseline);
assert(fp_tuned != fp_baseline);
```

Python (pytest):
```python
# Config sensitivity via output comparison
result_default = run_with_config({"sensitivity": 0.8})
result_zero    = run_with_config({"sensitivity": 0.0})
assert result_default != result_zero

# Scaled parameter changes output
monitor_full  = build_from_config({"scale": 1.0})
monitor_half  = build_from_config({"scale": 0.5})
assert monitor_half.effective_sensitivity < monitor_full.effective_sensitivity
```

## Checklist: Writing a New Test

Use this checklist whenever adding or reviewing a CI test.

### For any new feature or config key

- [ ] **Golden anchor exists**: at least one test asserts a fixed expected value
      for the default/baseline configuration.
- [ ] **Sensitivity probe exists**: at least one test changes the new parameter
      and asserts the output differs from the baseline.
- [ ] **Reproducibility assertion**: running the same config twice yields the
      same result (catches uninitialized memory, unseeded RNG, race conditions).
- [ ] **Seed is pinned**: every test that involves randomness sets an explicit
      seed.
- [ ] **Outcome assertion, not crash assertion**: the test checks a meaningful
      numerical or structural property, not just `!= nullptr` or `is not None`.

### For existing test suites (audit)

When reviewing or extending an existing test file, check:

- [ ] Does the suite have at least one golden-value test per major output?
- [ ] Does the suite have at least one sensitivity test per user-facing config
      key that is expected to alter the output?
- [ ] Are the golden values narrow enough to catch real drift but wide enough to
      survive legitimate precision changes across compilers/platforms?
- [ ] If a config key is parsed but has no sensitivity test, is there a comment
      explaining why (e.g., cosmetic-only key, output-format key)?

## Anti-Patterns to Avoid

| Anti-pattern | Why it's bad | Fix |
|---|---|---|
| Test only asserts no crash | Dead wiring is invisible | Add golden value + sensitivity assertions |
| Golden value with wide tolerance (>10%) | Drift goes undetected | Tighten to 1% or use exact match with seed pinning |
| Sensitivity test compares configs that differ in 5 ways | Can't isolate which parameter matters | Change exactly one parameter per comparison |
| Config parsed but no sensitivity test | Key can be silently ignored forever | Add a sensitivity probe |
| Golden file updated without explanation | Hides regressions | Require commit message rationale when updating goldens |
| Fingerprint without reproducibility check | Hash collision or nondeterminism hides real failures | Always pair with same-config-twice assertion |

## Organizing Tests by Role

Structure test files so both roles are visible:

```
tests/
  test_<module>.cpp          # unit tests: golden values for individual functions
  test_smoke.cpp             # integration: golden values for end-to-end runs
  test_config_diversity.cpp  # sensitivity: config changes -> distinct fingerprints
```

or in Python:

```
tests/
  test_<module>.py           # unit: golden values per function
  test_golden_orchestrator.py  # golden regression for full pipeline
  test_config_sensitivity.py   # sensitivity probes for all config keys
```

Within a single test file, group by role using comments or test classes:

```python
class TestGoldenValues:
    """Regression anchors -- expected outputs for fixed inputs."""
    def test_default_config_produces_known_output(self): ...
    def test_reproducible_on_repeat(self): ...

class TestConfigSensitivity:
    """Propagation probes -- changing config changes output."""
    def test_mutation_rate_changes_diversity(self): ...
    def test_false_alarm_penalty_changes_survival(self): ...
```

## Applying to Existing Repositories

### C++ simulation repos (e.g., GutIBM)

- `sim_fingerprint.h` provides a deterministic hash of simulation state.
- `test_config_diversity.cpp` is the canonical sensitivity test: it runs
  simulations from distinct config fixtures and asserts distinct fingerprints.
- When adding a new config key, extend `test_config_diversity.cpp` with a
  case that toggles or adjusts the new key and asserts the fingerprint differs.
- Golden values live in individual `test_<module>.cpp` files as hardcoded
  assertions (agent counts, biomass values, analytical solutions).

### Python simulation repos (e.g., TattleTots, Crusher-to-the-Bridge)

- `test_golden_orchestrator.py` is the canonical golden-value regression:
  it runs a full pipeline and asserts exact summary/cost values.
- `test_smoke.py` validates emergent behaviors as golden properties (trophic
  depth > 2, population stability, extinction cascades).
- Sensitivity tests are often embedded in module-level test files (e.g.,
  `test_wearable_enhanced.py` tests `sensitivity=0.0` suppresses detections,
  `detection_sensitivity_scale=0.5` halves effective sensitivity).
- When adding a new config key, add both:
  1. A golden assertion for the default value in the module's test file.
  2. A sensitivity assertion showing the key changes the output.

### Domain adapter repos (e.g., Scrapiron, Coral_Key, Xylella)

- Same two-pillar principle applies to domain-specific config (grid size,
  vessel count, ignition probability, etc.).
- Integration tests should assert that the adapter's config propagates
  through to the engine outputs (not just that the adapter initializes
  without error).

## When to Update Golden Values

Golden values should change only when:

1. **Algorithm improvement**: a deliberate change to the computation.
2. **Bug fix**: the old golden was wrong.
3. **Precision change**: compiler/platform change shifts floating-point results.

In all cases:
- Update the golden in the **same commit** as the code change.
- State in the commit message **why** the golden changed and **what** the new
  value represents.
- Never update a golden to make a failing test pass without understanding
  **why** the value changed.
