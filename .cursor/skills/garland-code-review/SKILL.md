---
name: garland-code-review
description: Review GARLAND code changes for bugs, privacy protocol correctness, test gaps, and documentation accuracy. Use when reviewing PRs, auditing simulation logic, or running /code-review on this repository.
paths:
  - "src/**"
  - "tests/**"
  - "README.md"
  - "pyproject.toml"
  - "docs/**"
---

# GARLAND Code Review

## When to Use

- Reviewing pull requests
- Running `/code-review` on GARLAND
- Pre-merge audit of privacy or simulation logic

## Review Priorities

1. **Correctness** — zone IDs, zone-local detection, SEIR conservation
2. **Privacy protocol** — token → dilution → broadcast → response chain
3. **Metrics** — episode-granular FN/TN; attack counters synced
4. **Tests** — regression tests for bug fixes; meaningful bounds
5. **Documentation** — README vs implementation
6. **Scope** — minimal diff

## High-Risk Checks

| Area | Question |
|------|----------|
| Zone IDs | Tokens, dilution, queries all use grid cell IDs? |
| Detection | TP/FP uses zone-local ground truth (not global timestep/count)? |
| FEBRILE/MULTI_SYSTEM | Still global? Flag if not fixed (#25) |
| Epsilon | Linear sum vs adaptive composition consistent with README? (#24) |
| Attacks | CLI flag → simulation hook → summary metric? |
| Tests | Regression test for bug fixes? |

## Open Bugs (do not re-introduce closed fixes)

| Issue | Type | Topic |
|-------|------|-------|
| #25 | Bug | FEBRILE/MULTI_SYSTEM global disease check |
| #24 | Bug | Linear ε vs adaptive composition |

**Closed — do not regress:** #2–#12, #16 (zone IDs, attacks, metrics, scaling, etc.)

Full list: `../garland-issues/references/known-issues.md`

## Documentation Red Flags

- NeuroKit2 "integration" (custom NumPy only unless #33 done)
- H3 indexing (rectangular grid unless #32 done)
- Formal DP proofs (simulated protocol — see #35)
- "Garlic routing" as implemented mesh protocol (broadcast-and-filter only)
- Replay attack missing from README attack bullets (#39)

## Test Quality

- [ ] Bug fix has regression test
- [ ] Assertions bound values, not `is not None` alone
- [ ] Integration tests use cell ID namespace
- [ ] Tests use `small_config` / reduced scale, not 250K in CI

## Current Test Suite

- **110 tests**, ~91% coverage
- Files: `test_simulation`, `test_privacy`, `test_attacks`, `test_cli`, `test_metrics`, `test_scaling`
- CI: `.github/workflows/tests.yml` (pytest; ruff not yet in CI — #31)

## Severity Guide

| Severity | Examples |
|----------|----------|
| **High** | Zone ID mismatch, broken protocol matching, attack flag no-op |
| **Medium** | Misleading metrics (#24, #25), doc/code mismatch |
| **Low** | Dead code (#27), encapsulation (#26), missing lint CI (#31) |

## Output Format

1. Summary (1–2 sentences on current quality)
2. Findings by severity with file references
3. What's working well
4. Suggested priority (bugs before features)

## Related Skills

- `garland-architecture` — system context
- `garland-privacy-protocol` — protocol specifics
- `garland-issues` — backlog and issue types
