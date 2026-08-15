---
name: sonar-quality
description: Prevent SonarCloud and Ruff quality issues when writing or changing Garland code.
---

# Garland Sonar Quality Standards

Apply these rules when writing or changing code in this repository. The SonarQube
project is `bckirkup_Garlic-Routed-Local-Area-Network-Domain`, with analysis wired
through `.github/workflows/tests.yml`.

## Local validation

```bash
pre-commit run --all-files
python scripts/sonar_guard.py src tests
python scripts/sonar_guard.py --workflows .github/workflows
uv sync --frozen --no-build --no-install-project --extra dev
uv run --no-sync --no-build ruff check src tests
uv run --no-sync --no-build ruff format --check src tests
uv run --no-sync --no-build mypy src/garland
PYTHONPATH=src uv run --no-sync --no-build python -m pytest tests/ -v
```

## Rule catalog

- `python:S3776`: existing functions may reach cognitive complexity 23. Do not
  refactor those legacy functions for this propagation; new or changed functions
  must remain below the local Ruff threshold of 15.
- `python:S7504`: six existing `list()` calls occur in the step pipeline. Some
  protect iteration from mutation, so leave them unchanged unless their behavior
  is reviewed separately.
- `pythonsecurity:S8707`: path validation must use an analyzer-visible
  ``realpath`` + ``startswith(base + os.sep)`` guard (see `src/garland/paths.py`).
  Do not reintroduce a boolean helper that hides the containment check from
  taint analysis.
- `python:S1244`: do not compare floats with ``==`` / ``!=``; use
  ``np.isclose`` / ``pytest.approx`` (or non-equality checks such as
  ``np.any(array)`` when exact zeros are expected).
- `python:S5778`, `python:S3358`, `python:S107`, and `python:S1172`: existing
  findings require targeted review and are not blanket exemptions.
- `githubactions:S8541`: published-package `pip install` commands must include
  `--only-binary :all:`.
- `githubactions:S8544`: published-package installs must use explicit versions,
  immutable commit references, or hashes.
- UV workflow commands require `--no-build`; `uv sync` additionally requires
  `--locked` or `--frozen`, and `uv run` should use `--no-sync --no-build`.
- `zizmor` separately checks that GitHub Actions references are immutable SHA pins.

## CI complexity ratchet

The whole-repository Ruff C901 ceiling is currently 16, measured from
`src/garland/app.py`. The pre-commit Ruff hook applies the goal threshold of 15
to changed Python files. The ceiling only ratchets downward.

## Ruff argument findings

Ruff enables `ARG` and `C901`. The ten existing `ARG002` findings are covered by
narrow documented per-file entries for interface-required parameters in
`src/garland/metrics.py`, `src/garland/simulation.py`, and the affected test
modules. Do not use `# noqa`, `nosonar`, or broad ignores. The existing `N806`
plume coefficient is intentionally not enabled by the Ruff selection.

## Known server-side findings

SonarCloud remains authoritative for interprocedural and taint analysis. Current
legacy findings include the S3776, S7504, S5778, S3358, S107, and S1172 classes
listed above. Fix only a narrowly mechanical issue that clearly preserves
behavior; escalate design decisions rather than weakening checks.
