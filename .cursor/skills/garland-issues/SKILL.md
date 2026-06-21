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

## Issue Types

Every issue should declare its type in the body:

| Type | GitHub label | When to use |
|------|--------------|-------------|
| **Bug fix** | `bug` | Incorrect behavior, misleading metrics, broken logic |
| **Enhancement** | `enhancement` | Cleanup, refactor, CI tooling, small improvements |
| **Documentation** | `documentation` | README, docstrings, skills — no logic change |
| **Feature** | `enhancement` | New capability; may need design discussion |

Include at top of issue body: `## Type\n**Bug fix**` (or Enhancement / Documentation / Feature).

## CLI Commands

```bash
# Open issues only
gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open

# View issue
gh issue view 25 -R bckirkup/Garlic-Routed-Local-Area-Network-Domain

# Create bug
gh issue create -R bckirkup/Garlic-Routed-Local-Area-Network-Domain \
  --title "Short title" \
  --label "bug" \
  --body "$(cat <<'EOF'
## Type
**Bug fix**
## Summary
...
## Acceptance criteria
- [ ] ...
EOF
)"
```

## Open Backlog (current)

### Bug fixes — fix first

| Priority | Issue | Topic |
|----------|-------|-------|
| P1 | [#25](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/25) | FEBRILE/MULTI_SYSTEM global disease classification |
| P1 | [#24](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/24) | Linear ε sum vs adaptive composition in summary |

### Enhancements — cleanup / CI

| Issue | Topic |
|-------|-------|
| [#27](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/27) | Remove dead `MaliciousAgent` |
| [#26](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/26) | Public `SpatialGrid` cell ID accessor |
| [#31](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/31) | Ruff in CI |
| [#37](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/37) | Type checking in CI |

### Documentation

| Issue | Topic |
|-------|-------|
| [#35](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/35) | Privacy guarantees as design intent |
| [#39](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/39) | plot_metrics docstring; replay in README |
| [#28](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/28) | CONTRIBUTING.md + CHANGELOG.md |

### Features — see feature backlog

| Issue | Topic |
|-------|-------|
| [#29](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/29) | Agent mobility |
| [#32](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/32) | H3 hex indexing |
| [#36](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/36) | YAML/TOML config |
| [#30](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/30) | Parameter sweeps |
| [#34](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/34) | Docker |
| [#33](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/33) | NeuroKit2 / OpenWearables |
| [#38](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/38) | Multi-plume scenarios |

**Full catalogs:**
- Bugs + closed history: `references/known-issues.md`
- Feature roadmap: `references/feature-backlog.md`

## Workflow for Fixing an Issue

1. Read issue — confirm **Type** and acceptance criteria
2. Branch: `git checkout -b cursor/fix-febrile-zone-b383`
3. Minimal fix matching existing conventions
4. Regression test (required for **Bug fix**)
5. `python3 -m pytest tests/ -v` and `ruff check src tests`
6. Commit: `Use zone-local disease check for febrile detections (closes #25)`
7. PR body: `Closes #25`

## Closed Issues (do not regress)

Issues #2–#12 and #16 are resolved on `main`. See `references/known-issues.md` for the full closed list.

## PR Template

```markdown
## Summary
Brief description.

Closes #25

## Type
Bug fix

## Test plan
- [ ] `python3 -m pytest tests/ -v`
- [ ] `ruff check src tests`
```

## References

- `references/known-issues.md` — open bugs + closed history
- `references/feature-backlog.md` — feature roadmap
- `../garland-testing/SKILL.md` — test requirements
- `../garland-architecture/SKILL.md` — system context
