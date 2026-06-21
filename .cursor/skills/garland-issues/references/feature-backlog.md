# GARLAND Feature Backlog

Suggested features not yet implemented. Each has a GitHub issue with **Type: Feature**.

Prioritize based on research goals — not all items are required for testbed use.

---

## Simulation realism

| Issue | Feature | Notes |
|-------|---------|-------|
| [#29](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/29) | Agent mobility | Positions static today; requires cell index rebuild |
| [#38](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/38) | Multi-plume / multi-outbreak | Single plume + single SEIR wave only |
| — | Structured environments | Schools, hospitals, transit (not filed — fold into mobility) |
| — | Epidemiological calibration | Validate SEIR params against real outbreak data (not filed) |

---

## Spatial & indexing

| Issue | Feature | Notes |
|-------|---------|-------|
| [#32](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/32) | H3 hex indexing | README lists as planned; rectangular grid only today |

---

## Biometrics & sensing

| Issue | Feature | Notes |
|-------|---------|-------|
| [#33](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/33) | NeuroKit2 / OpenWearables | Custom NumPy synthesis today |
| — | Wearable dropout / sensor failure | Not filed |
| — | Baseline warm-up period | Not filed |

---

## Privacy & security (research depth)

| Issue | Feature | Notes |
|-------|---------|-------|
| — | Real homomorphic encryption | Tokens are plaintext tuples (simulated protocol) |
| — | Formal privacy analysis / audit | Design goals only in README |
| — | Mesh / garlic routing layer | Name is conceptual; broadcast-and-filter only |
| — | Byzantine aggregator models | Eclipse drops tokens; no full adversarial aggregator |

---

## Experiment tooling

| Issue | Feature | Notes |
|-------|---------|-------|
| [#36](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/36) | YAML/TOML config files | CLI-only configuration today |
| [#30](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/30) | Parameter sweep runner | Compare ε vs detection trade-offs |
| [#34](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/34) | Docker environment | Reproducible city-scale runs |
| — | Interactive dashboard | CSV/PNG only today |

---

## Engineering

| Issue | Feature | Notes |
|-------|---------|-------|
| [#31](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/31) | Ruff in CI | Tests only in CI today |
| [#37](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/37) | Type checking in CI | Hints present, no mypy/pyright |
| — | Coverage fail threshold | Coverage reported, not enforced |
| [#28](https://github.com/bckirkup/Garlic-Routed-Local-Area-Network-Domain/issues/28) | CONTRIBUTING + CHANGELOG | Contributor onboarding |

---

## Filing new features

Use label `enhancement` and include in body:

```markdown
## Type
**Feature**

## Summary
...

## Acceptance criteria
- [ ] ...
```

For bugs, use label `bug` and `## Type\n**Bug fix**`.
