# GARLAND Agent Skills

Cursor Agent Skills for [GARLAND](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain) — a privacy-preserving epidemiological security testbed.

Invoke manually in Agent chat with `/skill-name`, or let skills load automatically when relevant.

## Skills

| Skill | Command | Use when |
|-------|---------|----------|
| [garland-development](garland-development/SKILL.md) | `/garland-development` | Setup, CLI, lint, git, PR workflow |
| [garland-testing](garland-testing/SKILL.md) | `/garland-testing` | pytest, coverage, writing tests |
| [garland-issues](garland-issues/SKILL.md) | `/garland-issues` | Triage, file, or close GitHub issues |
| [garland-architecture](garland-architecture/SKILL.md) | `/garland-architecture` | Navigate modules and data flow |
| [garland-privacy-protocol](garland-privacy-protocol/SKILL.md) | `/garland-privacy-protocol` | Tokens, dilution, DP, detection |
| [garland-code-review](garland-code-review/SKILL.md) | `/garland-code-review` | PR review and `/code-review` |

## Reference catalogs

| File | Purpose |
|------|---------|
| [resolved-issues.md](garland-issues/references/resolved-issues.md) | Closed bugs — do not regress |
| [feature-backlog.md](garland-issues/references/feature-backlog.md) | Future features and research directions |

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v
ruff check src tests
python -m garland.benchmark --quick
```

## Project snapshot

- **Stack:** Python ≥ 3.10, NumPy, Mesa (thin), pytest, ruff
- **Scale:** 250K agents default; 110 tests, ~91% coverage
- **CI:** `.github/workflows/tests.yml` (Python 3.10 & 3.12)
- **Docs:** `README.md`, `docs/SCALING.md`

Check open GitHub issues: `gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open`
