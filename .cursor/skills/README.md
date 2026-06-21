# GARLAND Agent Skills

Cursor Agent Skills for the [GARLAND](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain) epidemiological security testbed.

Skills are loaded automatically when relevant, or invoke manually with `/skill-name` in Agent chat.

| Skill | Invoke | Purpose |
|-------|--------|---------|
| [garland-development](garland-development/SKILL.md) | `/garland-development` | Setup, CLI, lint, git workflow |
| [garland-testing](garland-testing/SKILL.md) | `/garland-testing` | pytest, fixtures, coverage |
| [garland-issues](garland-issues/SKILL.md) | `/garland-issues` | GitHub issue triage and fixes |
| [garland-architecture](garland-architecture/SKILL.md) | `/garland-architecture` | Codebase structure and data flow |
| [garland-privacy-protocol](garland-privacy-protocol/SKILL.md) | `/garland-privacy-protocol` | DP protocol and spatial zones |
| [garland-code-review](garland-code-review/SKILL.md) | `/garland-code-review` | PR and code review checklist |

## Quick Start

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

See `garland-development` for full setup and `garland-issues/references/known-issues.md` for the open backlog (#2–#12).
