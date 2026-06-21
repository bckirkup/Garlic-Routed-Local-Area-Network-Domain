# GARLAND Agent Skills

Cursor Agent Skills for [GARLAND](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain).

Invoke with `/skill-name` in Agent chat, or let skills load automatically when relevant.

## Skills

| Skill | Command | Use when |
|-------|---------|----------|
| [garland-development](garland-development/SKILL.md) | `/garland-development` | Setup, CLI, config files, lint, git, PRs |
| [garland-testing](garland-testing/SKILL.md) | `/garland-testing` | pytest, mypy, coverage, writing tests |
| [garland-issues](garland-issues/SKILL.md) | `/garland-issues` | Triage, file, or close GitHub issues |
| [garland-architecture](garland-architecture/SKILL.md) | `/garland-architecture` | Modules, data flow, spatial/mobility |
| [garland-privacy-protocol](garland-privacy-protocol/SKILL.md) | `/garland-privacy-protocol` | Tokens, dilution, DP, detection |
| [garland-code-review](garland-code-review/SKILL.md) | `/garland-code-review` | PR review and `/code-review` |

## Reference catalogs

| File | Purpose |
|------|---------|
| [resolved-issues.md](garland-issues/references/resolved-issues.md) | Closed bugs — regression checklist |
| [feature-backlog.md](garland-issues/references/feature-backlog.md) | Remaining research / product ideas |

## Quick start

```bash
pip install -e ".[dev,biosignals]"   # optional NeuroKit2 path
python -m pytest tests/ -v
ruff check src tests
mypy
garland --config examples/quick.yaml
garland sweep --sweep-config examples/privacy_sweep.yaml
```

## Project snapshot

- **168 tests**, coverage enforced in CI
- **CI:** ruff + mypy + pytest (Python 3.10 & 3.12)
- **Spatial:** H3 hex default; rectangular via `--spatial-backend rect`
- **Config:** YAML/TOML files + CLI overrides; sweeps via `garland sweep`
- **Docs:** `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/SCALING.md`, `docs/BIOMETRICS.md`

Open issues: `gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open`
