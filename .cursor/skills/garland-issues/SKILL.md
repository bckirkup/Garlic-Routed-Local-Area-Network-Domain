---
name: garland-issues
description: Triage, implement, and close GitHub issues for GARLAND. Use when fixing bugs, resolving open issues, writing issue descriptions, linking PRs to issues, or prioritizing work from the backlog.
---

# GARLAND Issue Handling

## When to Use

- Picking up work from the GitHub issue backlog
- Implementing a fix for a filed bug
- Creating new issues for regressions or scope decisions
- Writing PR descriptions that reference issues

## Repository

- **GitHub:** `bckirkup/Garlic-Routed-Local-Area-Network-Domain`
- **Default branch:** `main`
- **Feature branches:** `cursor/<descriptive-name>-b383`

## CLI Commands

```bash
# List open issues
gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain

# View issue details
gh issue view 5 -R bckirkup/Garlic-Routed-Local-Area-Network-Domain

# Create issue
gh issue create -R bckirkup/Garlic-Routed-Local-Area-Network-Domain \
  --title "Short title" \
  --label "bug" \
  --body "$(cat <<'EOF'
## Summary
...
## Acceptance criteria
- [ ] ...
EOF
)"

# Close when PR merges (in PR body)
# Closes #5
```

## Available Labels

| Label | Use for |
|-------|---------|
| `bug` | Incorrect behavior, broken install, integration failures |
| `documentation` | README, license, doc/code mismatches |
| `enhancement` | New features, deps cleanup, benchmarks, test improvements |
| `good first issue` | Small, well-scoped fixes |
| `help wanted` | Needs design input or larger effort |

## Priority Order (recommended)

Fix in this order unless the user specifies otherwise:

| Priority | Issue | Topic |
|----------|-------|-------|
| P0 | #5 | Zone ID namespace mismatch (breaks privacy protocol) |
| P0 | #3 | Missing `networkx` dependency (broken fresh install) |
| P1 | #4 | Attack layer not integrated |
| P1 | #7 | Dead metrics fields in summary |
| P1 | #2 | Detection classification uses global plume timing |
| P2 | #8 | FNR inflated by per-step counting |
| P2 | #9 | Household/wearable spatial model mismatch |
| P2 | #6 | License inconsistency (MIT vs Apache) |
| P3 | #10 | Unused dependencies |
| P3 | #11 | Performance validation at 250K scale (resolved) |
| P3 | #12 | Test coverage gaps |

Full details: `.cursor/skills/garland-issues/references/known-issues.md`

## Workflow for Fixing an Issue

1. **Read the issue** — confirm acceptance criteria and affected files
2. **Create branch** — `git checkout -b cursor/fix-zone-id-b383` (from `main`)
3. **Implement minimal fix** — smallest correct diff; match existing conventions
4. **Add regression test** — see `garland-testing` skill
5. **Run tests and lint** — `python3 -m pytest tests/ -v`
6. **Commit** — `Fix zone ID mismatch in privacy protocol (closes #5)`
7. **Push and open PR** — body includes `Closes #N`
8. **Verify acceptance criteria** — check off items in PR description

## Creating New Issues

Use this template:

```markdown
## Summary
One paragraph describing the problem.

## Affected code
- `src/garland/...`

## Impact
What breaks or misleads users/researchers.

## Suggested fix
Concrete approach (optional).

## Acceptance criteria
- [ ] Measurable outcome 1
- [ ] Regression test added
```

**When to file vs fix inline:**

- File when scope is uncertain, needs design decision, or spans multiple PRs
- Fix inline without issue only for typos or trivial one-line fixes the user requested directly

## Scope Decisions (issues #4, #10)

Some issues allow two valid resolutions:

| Issue | Option A | Option B |
|-------|----------|----------|
| #4 Attacks | Wire all attacks into simulation | Remove CLI flags and README claims until ready |
| #10 Unused deps | Remove unused packages | Implement advertised integrations (NeuroKit2, H3) |

**Default:** prefer minimal fix (wire or remove claims) unless the user asks for full feature implementation.

## PR ↔ Issue Linking

Always include in PR body:

```markdown
## Summary
Fixes zone ID mismatch so tokens and dilated zones use grid cell IDs.

Closes #5

## Test plan
- [ ] `python3 -m pytest tests/ -v`
- [ ] New integration test `test_broadcast_matches_cell_zone`
```

## References

- Known issue catalog: `references/known-issues.md`
- Architecture context: `../garland-architecture/SKILL.md`
- Testing requirements: `../garland-testing/SKILL.md`
