# Contributing to GARLAND

Thank you for contributing to the Garlic-Routed Local Area Network Domain (GARLAND) epidemiological security testbed.

## Development setup

```bash
git clone https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain.git
cd Garlic-Routed-Local-Area-Network-Domain
pip install -e ".[dev]"
```

Verify your environment:

```bash
python -m pytest tests/ -v
ruff check src tests
mypy
garland --n-agents 1000 --n-steps 48 --no-plots
```

## Branch naming

Create feature branches from `main`:

```text
cursor/<descriptive-name>-<suffix>
```

Use lowercase and hyphens. Cloud agents typically append a short suffix (e.g. `-9c3a`).

## Making changes

1. **Read the relevant issue** — confirm acceptance criteria before starting.
2. **Keep diffs focused** — match existing conventions in `src/garland/` (dataclasses, NumPy vectorization, module docstrings).
3. **Add regression tests** for bug fixes and new behavior.
4. **Run the full check suite** before pushing:

   ```bash
   ruff check src tests
   mypy
   python -m pytest tests/ -v
   ```

5. **Commit with clear messages** — complete sentences describing what changed and why.
6. **Link issues in PRs** — include `Closes #N` in the PR body so issues auto-close on merge.

## Writing tests

Example-based tests live under `tests/`. For privacy primitives and config parsing, we also use [Hypothesis](https://hypothesis.readthedocs.io/) property tests in `tests/test_property.py`. These generate many random inputs to catch edge cases while keeping `max_examples` low for fast CI runs.

When adding property tests:

- Assert invariants (finite outputs, round-trips, rejection of invalid types), not brittle exact values.
- Use `@settings(max_examples=30)` or similar so `pytest tests/ -v` stays quick.
- Seed NumPy generators when statistical checks need reproducibility.

## Pull request expectations

- Draft PRs are fine for work in progress.
- PR description should summarize changes and list the test plan.
- CI must pass (lint, type check, tests on Python 3.10 and 3.12).
- Do not add unused dependencies or drive-by refactors unrelated to the issue.

## Project layout

| Path | Purpose |
|------|---------|
| `src/garland/` | Simulation engine, privacy protocol, CLI |
| `tests/` | Pytest suite |
| `.cursor/skills/` | Agent skills for development, testing, and architecture |
| `output/` | Simulation outputs (gitignored) |

## Getting help

- Open a [GitHub issue](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues) for bugs or feature requests.
- See `README.md` for architecture overview and CLI usage.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
