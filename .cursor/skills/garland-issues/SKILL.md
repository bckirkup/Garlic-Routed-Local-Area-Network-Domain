---
name: garland-issues
description: Triage, file, and close GitHub issues for GARLAND. Use when working the backlog, fixing bugs, scoping features, or linking PRs with Closes #N.
---

# GARLAND Issue Handling

## Repository

- **GitHub:** `bckirkup/Garlic-Routed-Local-Area-Network-Domain`
- **Branch template:** `cursor/<descriptive-name>-b383`
- **Contributor guide:** `CONTRIBUTING.md`

## Issue Types

| Type | Label | Requirement |
|------|-------|-------------|
| **Bug fix** | `bug` | Regression test |
| **Enhancement** | `enhancement` | CI/tests pass |
| **Documentation** | `documentation` | Docs only |
| **Feature** | `enhancement` | Tests + docs |

Include in body: `## Type\n**Bug fix**`

## CLI

```bash
gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open
gh issue view 25 -R bckirkup/Garlic-Routed-Local-Area-Network-Domain
```

**Skills are not the source of truth for open issues** — always check GitHub.

## Catalogs

| File | Use |
|------|-----|
| `references/resolved-issues.md` | Regression checklist (#2–#39) |
| `references/feature-backlog.md` | Stretch goals still open |
| `CHANGELOG.md` | What shipped on main |

## Workflow

1. Confirm issue open + acceptance criteria
2. `git checkout main && git pull origin main`
3. `git checkout -b cursor/fix-desc-b383`
4. Fix + test (`ruff`, `mypy`, `pytest`)
5. PR with `Closes #N`

## Priority

1. Bug fixes affecting metrics or privacy protocol
2. CI/regression failures
3. Documentation drift
4. Features from backlog / new issues

## PR Template

```markdown
## Summary
...

## Type
Bug fix

Closes #N

## Test plan
- [ ] ruff check src tests
- [ ] mypy
- [ ] python -m pytest tests/ -v
```
