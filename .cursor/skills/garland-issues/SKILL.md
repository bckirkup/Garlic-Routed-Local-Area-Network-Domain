---
name: garland-issues
description: Triage, file, implement, and close GitHub issues for GARLAND. Use when working the backlog, fixing bugs, scoping features, writing issue bodies, or linking PRs to issues.
---

# GARLAND Issue Handling

## When to Use

- Picking up work from the GitHub backlog
- Filing new bugs or feature requests
- Writing PR descriptions with `Closes #N`
- Prioritizing bug fixes vs features

## Repository

- **GitHub:** `bckirkup/Garlic-Routed-Local-Area-Network-Domain`
- **Default branch:** `main`
- **Branches:** `cursor/<descriptive-name>-b383`

## Issue Types

Tag every issue with a **Type** in the body and matching label:

| Type | Label | Meaning |
|------|-------|---------|
| **Bug fix** | `bug` | Incorrect behavior; needs regression test |
| **Enhancement** | `enhancement` | Cleanup, CI, refactor, tooling |
| **Documentation** | `documentation` | Docs only |
| **Feature** | `enhancement` | New capability |

```markdown
## Type
**Bug fix**

## Summary
...

## Acceptance criteria
- [ ] Regression test added
- [ ] `pytest tests/ -v` passes
```

## CLI

```bash
# Open issues
gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open

# View
gh issue view 25 -R bckirkup/Garlic-Routed-Local-Area-Network-Domain

# Create bug
gh issue create -R bckirkup/Garlic-Routed-Local-Area-Network-Domain \
  --title "Short title" --label "bug" --body-file issue.md

# Close via PR
# PR body: Closes #25
```

## Catalogs (in this repo)

| File | Contents |
|------|----------|
| `references/resolved-issues.md` | Closed bugs — regression checklist |
| `references/feature-backlog.md` | Future features and research ideas |

**Do not treat skill files as the source of truth for open issues.** Always run `gh issue list --state open` before starting work.

## Fix Workflow

1. Confirm issue is open and read acceptance criteria
2. `git checkout main && git pull origin main`
3. `git checkout -b cursor/fix-short-desc-b383`
4. Minimal fix + regression test (bugs)
5. `python3 -m pytest tests/ -v && ruff check src tests`
6. Commit: `Fix febrile zone-local classification (closes #25)`
7. Push; open PR with `Closes #25` and test plan

## PR Template

```markdown
## Summary
What changed and why.

## Type
Bug fix | Enhancement | Documentation | Feature

Closes #NN

## Test plan
- [ ] `python3 -m pytest tests/ -v`
- [ ] `ruff check src tests`
- [ ] New regression test: `test_...`
```

## Priority Guidance

1. **Bug fixes** that affect metrics correctness or privacy protocol behavior
2. **CI / lint** gaps that allow regressions to merge
3. **Documentation** drift
4. **Features** from `feature-backlog.md` based on research goals

## Scope Rules

- One issue per PR when possible
- Do not mix unrelated refactors with bug fixes
- If a feature needs design input, file issue first and use `help wanted`

## References

- `references/resolved-issues.md`
- `references/feature-backlog.md`
- `../garland-testing/SKILL.md`
- `../garland-code-review/SKILL.md`
