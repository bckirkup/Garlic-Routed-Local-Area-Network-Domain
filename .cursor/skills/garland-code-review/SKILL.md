---
name: garland-code-review
description: Review GARLAND PRs and code changes. Use for /code-review, pre-merge audit of simulation, privacy, spatial, config, and metrics logic.
paths:
  - "src/**"
  - "tests/**"
  - "README.md"
  - "CONTRIBUTING.md"
  - "CHANGELOG.md"
  - "docs/**"
---

# GARLAND Code Review

## Order

1. Correctness (protocol, zone IDs, detection, mobility cell sync)
2. Regressions (`resolved-issues.md`, `CHANGELOG.md`)
3. Tests + CI (`ruff`, `mypy`, pytest)
4. Docs vs behavior
5. Scope

## Checklist

| Area | Question |
|------|----------|
| Spatial | H3 and rect both work if dilution/index changed? |
| Mobility | Cell IDs rebuilt after move? |
| Detection | Zone-local instances for all anomaly types? |
| Multi-hazard | Instance IDs tracked in metrics? |
| Config | YAML/TOML + CLI override behavior preserved? |
| Attacks | Flag → hook → summary metric? |
| Tests | Regression test for bug fixes? |

## Do Not Regress

See `../garland-issues/references/resolved-issues.md` — especially #5 zone IDs, #8 episode metrics, #25 febrile zones.

## Red Flags

- Claiming formal DP proofs (simulation testbed)
- Real homomorphic encryption (not implemented)
- Mesh routing (broadcast-and-filter only)
- Breaking `garland sweep` or `--config` without tests

## CI Baseline

- `ruff check src tests`
- `mypy`
- 168 tests, coverage in pytest addopts

## Severity

| Level | Examples |
|-------|----------|
| High | Broken protocol matching, mobility desync, wrong TP logic |
| Medium | Misleading metrics/docs |
| Low | Style, minor refactors |

## Output Format

1. Summary
2. Findings by severity (code citations)
3. Strengths
4. Next steps

## References

- `CONTRIBUTING.md`
- `garland-architecture`, `garland-privacy-protocol`
