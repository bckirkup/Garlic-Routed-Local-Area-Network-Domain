# GARLAND Agent Skills

Cursor Agent Skills for the [GARLAND](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain) epidemiological security testbed.

Skills load automatically when relevant, or invoke manually with `/skill-name` in Agent chat.

| Skill | Invoke | Purpose |
|-------|--------|---------|
| [garland-development](garland-development/SKILL.md) | `/garland-development` | Setup, CLI, lint, git workflow |
| [garland-testing](garland-testing/SKILL.md) | `/garland-testing` | pytest, fixtures, coverage (~110 tests) |
| [garland-issues](garland-issues/SKILL.md) | `/garland-issues` | Issue triage, bug vs feature tagging |
| [garland-architecture](garland-architecture/SKILL.md) | `/garland-architecture` | Codebase structure and data flow |
| [garland-privacy-protocol](garland-privacy-protocol/SKILL.md) | `/garland-privacy-protocol` | DP protocol and spatial zones |
| [garland-code-review](garland-code-review/SKILL.md) | `/garland-code-review` | PR and code review checklist |

## Issue catalogs

| File | Contents |
|------|----------|
| [known-issues.md](garland-issues/references/known-issues.md) | Open bugs (#24, #25), enhancements, closed history |
| [feature-backlog.md](garland-issues/references/feature-backlog.md) | Feature roadmap (#29–#38 and suggestions) |

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v
python -m garland.benchmark --quick
```

## Current state (main)

- **110 tests**, ~91% coverage, CI on Python 3.10/3.12
- **Open bugs:** #24 (epsilon accounting), #25 (febrile zone classification)
- **Resolved:** zone IDs, attacks, metrics, scaling, deps (#2–#12, #16)
