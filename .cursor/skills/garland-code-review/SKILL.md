---
name: garland-code-review
description: Review GARLAND code changes for bugs, privacy protocol correctness, test gaps, and documentation accuracy. Use when reviewing PRs, auditing simulation logic, or running /code-review on this repository.
paths:
  - "src/**"
  - "tests/**"
  - "README.md"
  - "pyproject.toml"
---

# GARLAND Code Review

## When to Use

- Reviewing pull requests in this repository
- Running `/code-review` on GARLAND changes
- Pre-merge audit of privacy or simulation logic

## Review Priorities (in order)

1. **Correctness** — especially zone ID consistency, detection logic, SEIR conservation
2. **Privacy protocol integrity** — token → dilution → broadcast → response chain
3. **Behavioral regressions** — metrics, attack wiring, CLI flags
4. **Test coverage** — regression tests for bug fixes; meaningful assertions
5. **Documentation accuracy** — README claims vs implementation
6. **Scope** — no unrelated refactors or unused dependencies

## High-Risk Areas

Check these on every privacy/simulation PR:

| Area | Question |
|------|----------|
| Zone IDs | Do tokens, dilution, and query matching use the same spatial namespace (grid cell IDs)? |
| Detection | Is TP/FP classification based on zone-local ground truth, not global timestep? |
| Metrics | Are summary fields actually updated? |
| Attacks | If CLI flag exists, does simulation loop invoke the attack? |
| Tests | Is there a regression test, not just smoke "runs without error"? |

## Known Open Issues

Review against backlog — do not re-introduce fixed bugs:

See `.cursor/skills/garland-issues/references/known-issues.md` (issues #2–#12).

## Documentation Red Flags

README overclaims — flag if PR adds docs referencing:

- NeuroKit2 integration (custom NumPy biometrics only)
- H3 hex indexing (rectangular grid only)
- Full attack suite (Sybil only wired in simulation)
- Formal DP proofs (simulated protocol only)

## Test Quality Checks

- [ ] Bug fix includes regression test
- [ ] Assertions bound values, not just `is not None`
- [ ] No silent skip via `if x is not None` without guaranteed fixture
- [ ] Integration tests use same ID namespace as production code
- [ ] Tests use `small_config` scale, not 250K agents

## Dependency Checks

- [ ] New deps are actually imported and used
- [ ] Transitive deps (e.g. `networkx` for Mesa) declared in pyproject
- [ ] No unnecessary weight (`scipy`, `pydantic`) without justification

## Style (secondary)

- Dataclasses for config; NumPy for scale
- Line length 100; Ruff E/F/W/I
- Module docstrings present
- Minimal diff scope

## Severity Labels for Findings

| Severity | Examples |
|----------|----------|
| **High** | Zone ID mismatch, broken install, attack flag no-op, wrong privacy behavior |
| **Medium** | Misleading metrics, doc/code mismatch, FNR inflation |
| **Low** | Unused deps, missing benchmarks, weak test assertions |

## Output Format

Structure review findings as:

1. Summary (1–2 sentences)
2. Findings by severity with file references
3. What's working well
4. Suggested fix order (if applicable)

Use code citation format: ` ```startLine:endLine:filepath ` `

## Related Skills

- `garland-architecture` — system context
- `garland-privacy-protocol` — protocol specifics
- `garland-testing` — test expectations
