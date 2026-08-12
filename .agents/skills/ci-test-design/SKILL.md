---
name: ci-test-design
description: Design CI tests that actually test behavior — graded sensitivity ("a few different values here produce a few different values there") and bounds/invariants (numbers stay in range, finite, conserved) — with golden values demoted to a small number of labeled change-detectors. Use when writing or reviewing any test for any repository.
---

# CI Test Design: Sensitivity + Bounds First, Goldens Last

The default failure mode of this codebase's test suites is **too many golden
tests, too early**. A golden test (snapshot hash, "determinism fingerprint",
`assert cost == 0.4271`) is cheap to write and weak as a test:

- It fails on **every** legitimate change, so it gets updated reflexively rather
  than understood.
- It passes on numbers that are **stable but wrong** — a dead config knob, an
  inverted sign, a 1000x unit error all reproduce perfectly.
- `fp_changed != fp_baseline` proves a parameter reached *something*. It does not
  prove it reached the *right* thing, in the right direction, with a meaningful
  magnitude.

So the priority order for any mechanism under test is:

1. **Graded sensitivity** — a few different values here produce a few different
   values there, ordered the way the science says, by a non-trivial amount.
2. **Bounds and invariants** — numbers do not go far out of bounds, go NaN, or
   silently violate a conservation law.
3. **Golden values** — at most one or two per pipeline, explicitly labeled as
   change-detectors, not correctness checks.

A suite with (1) and (2) and no goldens is healthy. A suite with only goldens is
a tripwire, not a test suite.

## 1. Graded sensitivity tests (the primary tool)

Sweep one input over **three or more values** and assert on the *shape of the
response*, not on point values.

Assert as many of these as apply:

- **Ordering / monotonicity**: the metric moves the way the domain says it must
  across the whole sweep (higher toxin dose -> survivors non-increasing; higher
  trust -> desertion probability non-increasing).
- **Distinctness with a margin**: outputs differ by more than noise. Never
  `assert a != b` on floats — require a stated minimum separation.
- **Non-degeneracy (live knob)**: the response spans a non-trivial fraction of
  the output scale. A knob whose 10x change moves the metric <1% is a dead knob;
  the test should fail and that is a finding, not a test to loosen.
- **Saturation is asserted deliberately, not accidentally**: if a response is
  expected to plateau, assert the plateau *and* assert the pre-plateau region is
  responsive.
- **Negative control**: perturbing an *unrelated* parameter leaves the metric
  (near-)unchanged. Without this, "it changed" tests pass on global chaos.

Shape:

```python
def test_kill_rate_grades_survivors():
    values = [0.0, 1e-4, 1e-3, 1e-2]          # 3+ points, wide span
    metric = [run(seed=7, kill_rate=v).survivors for v in values]

    assert metric == sorted(metric, reverse=True)          # ordering
    span = (max(metric) - min(metric)) / max(metric[0], 1)
    assert span > 0.2, f"kill_rate looks dead: span={span:.3f}"   # live knob
    assert min(metric) >= 0                                # bounds (see below)

def test_unrelated_key_does_not_move_survivors():
    base = run(seed=7).survivors
    other = run(seed=7, output_precision=3).survivors
    assert other == pytest.approx(base, rel=1e-9)          # negative control
```

C++ (CTest) — same structure, asserting on real observables rather than a hash:

```cpp
std::vector<double> doses{0.0, 5e-4, 5e-3};
std::vector<std::size_t> survivors;
for (double d : doses) {
  auto cfg = baseline; cfg.receptor.kill_rate_colicin = d;
  survivors.push_back(run(cfg).agents().size());
}
assert(std::is_sorted(survivors.rbegin(), survivors.rend()));       // ordering
assert(survivors.front() - survivors.back() > survivors.front()/5); // live knob
```

TypeScript (Vitest):

```ts
const taus = [0.1, 0.4, 0.7, 0.95];
const refusals = taus.map((t) => runMatch({ seed: 1, tauBenev: t }).refusals);
expect(refusals).toEqual([...refusals].sort((a, b) => b - a));  // ordering
expect(refusals[0] - refusals[refusals.length - 1]).toBeGreaterThan(2); // margin
```

**Rules**

1. One parameter varies per sweep; everything else, including the seed, held fixed.
2. Pick values spanning a wide range (0, default, 10x default), not 1.01x.
3. For stochastic systems, average over N seeds and compare distributions or
   means with a margin; state the N and why it suffices.
4. Assert on a **domain-meaningful metric** (survivors, burned area, detections,
   epsilon spent), never on a hash — a hash cannot be ordered.
5. Cover every user-facing config key expected to alter output. A key with no
   sensitivity test needs a comment saying why (cosmetic, output-format only).

## 2. Bounds and invariant tests

Cheap, durable, and they catch the failures goldens hide.

- **Domain ranges**: probabilities in [0,1]; concentrations, populations, counts
  >= 0; bounded quantities within their declared interval; populations <= carrying
  capacity.
- **Finiteness**: no NaN/Inf anywhere in state or output — checked after a
  realistic number of steps, not just at step 1.
- **Conservation / budget**: mass, energy, carbon, or privacy budget in ~= out +
  stored, within a stated tolerance, with the tolerance justified in a comment.
- **Stability / no blowup**: long-run values stay inside an order-of-magnitude
  envelope; no unbounded drift; a quiescent run stays quiescent.
- **Boundary inputs**: zero, max, empty population, single agent — no crash, and
  still in bounds.
- **Reproducibility where it is a feature**: same seed twice -> same result
  (catches unseeded RNG, uninitialized memory, iteration-order dependence).

Prefer property-based testing for this class where it is already available
(Hypothesis in `Garlic-Routed-Local-Area-Network-Domain`, fuzz tests in
`TheKingsAndI`): generate configs, assert invariants hold for all of them.

```python
@given(dose=floats(0, 1e-2), flow=floats(0, 5.0))
def test_state_stays_physical(dose, flow):
    s = run(seed=3, kill_rate=dose, advection=flow, steps=200).state
    assert np.all(np.isfinite(s.concentrations))
    assert np.all(s.concentrations >= 0)
    assert 0.0 <= s.detection_rate <= 1.0
    assert s.carbon_in == pytest.approx(s.carbon_out + s.carbon_stored, rel=1e-6)
```

## 3. Golden values (use sparingly)

Keep goldens only where the recorded value is genuinely the contract:

- seed-reproducibility / determinism IDs,
- an analytically known answer (compare to the closed-form solution, which is a
  correctness check, not a golden),
- one end-to-end change-detector per pipeline, named and commented as such.

If you keep one, it must say what it is:

```python
# CHANGE DETECTOR, not a correctness check. This value is not independently
# derived; it only pins current behavior. If a deliberate change moves it,
# update it and say why. It failing means "something moved", not "something broke".
GOLDEN_SUMMARY_COST = 0.4271
```

Never let a golden be the *only* coverage for a mechanism, and never add a
golden for a new config key in place of a sensitivity sweep.

## Checklist: new feature or config key

- [ ] **Sensitivity sweep**: 3+ values, ordering asserted, minimum effect size
      asserted (knob is provably alive).
- [ ] **Negative control**: an unrelated knob does not move the metric.
- [ ] **Bounds**: outputs stay in their domain range; nothing NaN/Inf after a
      realistic run length.
- [ ] **Invariant**: the relevant conservation/monotonicity/stability law is
      asserted with a justified tolerance.
- [ ] **Boundary inputs**: zero / max / empty handled and still in bounds.
- [ ] **Seed pinned** in every test involving randomness.
- [ ] **Metric is domain-meaningful**, not a hash and not `is not None`.
- [ ] **Goldens**: none added, or one labeled change-detector with a rationale.
- [ ] **Fast**: sweeps use reduced grid/steps/population so 3-5 runs are cheap.
      A sensitivity test too slow to run gets deleted, which is worse than none.

## Audit checklist for an existing suite

- [ ] For each mechanism, is there a graded response test, or only a golden?
- [ ] For each golden assertion: what was it protecting? Replace it with the
      sensitivity + bounds pair that actually protects that.
- [ ] Any `fp_a != fp_b` sensitivity test — can it be upgraded to an ordered
      sweep on a real metric?
- [ ] Any test whose only assertion is "no crash" or "not None"?
- [ ] Are there bounds/finiteness assertions at all, at realistic run lengths?
- [ ] Are there dead knobs — config keys with no test that they change anything?

## Anti-patterns

| Anti-pattern | Why it's bad | Fix |
|---|---|---|
| Golden hash as the primary test for a mechanism | Passes on stable-but-wrong numbers; fails on every real change | Ordered sensitivity sweep + bounds |
| `assert fp_changed != fp_baseline` | Proves only that *something* moved | Assert direction and magnitude on a real metric |
| Two-point comparison (`default` vs `zero`) | Cannot see ordering or saturation | Use 3+ values |
| `assert a != b` on floats | Passes on 1e-18 of numerical noise | Require a stated minimum separation |
| No negative control | "It changed" tests pass on global chaos | Perturb an unrelated knob, assert no movement |
| Bounds checked only at step 0/1 | Blowup and NaN appear later | Assert after a realistic run length |
| Loosening a sensitivity test to make it pass | Hides a dead knob — the real bug | Report the dead knob; fix the wiring |
| Updating a golden to make CI green | Hides regressions | Understand the move first; state why in the commit |
| Slow sweeps | Get skipped or deleted | Shrink grid/steps/population |

## Organizing tests by role

```
tests/
  test_<module>.{py,cpp,ts}     # unit: behavior, bounds, analytic checks
  test_sensitivity.{py,cpp,ts}  # graded sweeps per config key (+ negative controls)
  test_invariants.{py,cpp,ts}   # ranges, finiteness, conservation, stability
  test_smoke.{py,cpp,ts}        # end-to-end: runs, stays in bounds, 1 change-detector
```

Within a file, group by role:

```python
class TestSensitivity:
    """Graded response: a few different values in -> a few different values out."""

class TestInvariants:
    """Ranges, finiteness, conservation, stability."""

class TestChangeDetectors:
    """Pinned values. Not correctness checks — see comments."""
```

## Repo-specific notes

- **GutModelBacteriocins (C++/CTest)**: `sim_fingerprint.h` and
  `test_config_diversity.cpp` currently express sensitivity as fingerprint
  inequality. Keep them, but when touching a mechanism add an ordered sweep on a
  real observable (agent count, biomass, toxin flux) in the mechanism's own
  `test_<module>.cpp`, plus the carbon/mass budget invariant.
- **TheKingsAndI (TypeScript/Vitest)**: golden fingerprints are a determinism
  feature and can stay as change-detectors. Psychology mechanisms (tau, trauma,
  desertion thresholds) need graded sweeps asserting ordering, and bounds on
  trust/probability values.
- **TattleTots and domain adapters (Scrapiron, Coral_Key, Xylella, Garland)**:
  domain config (grid size, ignition probability, vessel count, sensor noise)
  gets an ordered sweep through the adapter into engine outputs — not just
  "adapter initializes". Energy/attention budgets get conservation invariants.
