---
name: garland-code-review
description: Review GARLAND pull requests and code changes. Use for PR review, /code-review, or auditing simulation, privacy, metrics, and attack logic.
paths:
  - "src/**"
  - "tests/**"
  - "README.md"
  - "docs/**"
  - "pyproject.toml"
---

# GARLAND Code Review

## Review Order

1. **Correctness** — protocol matching, zone IDs, detection ground truth, SEIR conservation
2. **Regressions** — see `../garland-issues/references/resolved-issues.md`
3. **Tests** — regression test for every bug fix; meaningful assertions
4. **Docs** — README claims vs implementation
5. **Scope** — minimal diff

## High-Risk Checklist

| Check | Question |
|-------|----------|
| Zone IDs | Tokens, dilution, queries all use grid **cell IDs**? |
| Detection | TP/FP uses zone-local plume/disease, not global timestep? |
| Metrics | Episode FN/TN (not per-step inflation)? Attack counters synced? |
| Attacks | CLI flag → simulation hook → summary field? |
| Privacy | Dummy tokens filtered? K-anonymity dilution before broadcast? |
| Tests | New regression test? No 250K agents in CI? |

## Regression Hotspots

Do not re-introduce fixes from resolved issues:

- Zone ID mismatch (tokens vs dilution)
- Global plume timing for toxin TP
- Unwired attack flags or dead summary metrics
- Per-step FN inflation
- Household/neighborhood spatial incoherence
- Missing `networkx` dependency

Full list: `../garland-issues/references/resolved-issues.md`

## Documentation Red Flags

| Claim | Reality |
|-------|---------|
| NeuroKit2 integration | Custom NumPy synthesis (NeuroKit2-inspired) |
| H3 hex grid | Rectangular cells; H3 is future work |
| Formal DP proofs | Simulated protocol; design goals in README |
| Garlic mesh routing | Broadcast-and-filter; no routing layer |
| Homomorphic encryption | Plaintext token tuples |

## Test Quality

- [ ] Bug fix has failing-then-passing regression test
- [ ] Bounds on numeric assertions (not bare `is not None`)
- [ ] No conditional skips without guaranteed fixtures
- [ ] Integration tests for protocol path changes

## Style (secondary)

- Dataclasses for config; NumPy at scale
- Ruff E/F/W/I; line length 100
- Module docstrings on public modules

## Severity Labels

| Level | Examples |
|-------|----------|
| **High** | Broken protocol matching, wrong zone IDs, attack no-op |
| **Medium** | Misleading metrics, doc/code mismatch |
| **Low** | Dead code, missing lint CI, encapsulation |

## Output Format

1. Brief quality summary
2. Findings by severity (with code citations)
3. What's working well
4. Suggested next steps

## Current Baseline

- 110 tests, ~91% coverage
- CI: pytest on Python 3.10/3.12
- Five attack types wired with CLI and metrics

## Related Skills

- `garland-architecture`
- `garland-privacy-protocol`
- `garland-testing`
