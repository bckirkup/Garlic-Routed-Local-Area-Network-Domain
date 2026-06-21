# GARLAND Feature Backlog

Future capabilities **not required** for core testbed use. Check GitHub for filed issues; file new ones with `## Type\n**Feature**`.

```bash
gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open --label enhancement
```

---

## Simulation realism

| Feature | Notes | Suggested issue |
|---------|-------|-----------------|
| **Agent mobility** | Positions static after init; cell cache would need rebuild | #29 |
| **Multi-plume / multi-outbreak** | Single plume source + single SEIR wave today | #38 |
| **Structured venues** | Schools, hospitals, transit hubs | New issue |
| **Epidemiological calibration** | Validate SEIR against real outbreak data | New issue |
| **SEIR fidelity at scale** | `max_infectious_checks=500` samples transmission sources | Configurable via `SEIRConfig` |

---

## Spatial & indexing

| Feature | Notes | Suggested issue |
|---------|-------|-----------------|
| **H3 hex grid** | Rectangular 200 m cells today | #32 |
| **Dynamic cell rebuild** | Prerequisite for mobility | #29 |

---

## Biometrics & sensing

| Feature | Notes | Suggested issue |
|---------|-------|-----------------|
| **NeuroKit2 / OpenWearables** | Custom NumPy synthesis today | #33 |
| **Sensor dropout / battery** | All wearables always on | New issue |
| **Baseline warm-up period** | Anomaly detection from step 0 | New issue |

---

## Privacy & security (research depth)

| Feature | Notes |
|---------|-------|
| **Real homomorphic encryption** | Tokens are plaintext simulation |
| **Adaptive composition in live metrics** | Function exists; ensure wired to output |
| **Formal privacy audit** | Design goals only in README |
| **Mesh / garlic routing layer** | Broadcast-and-filter; name is conceptual |
| **Byzantine aggregator** | Eclipse drops tokens; no full adversarial node model |

---

## Experiment tooling

| Feature | Notes | Suggested issue |
|---------|-------|-----------------|
| **YAML/TOML configs** | CLI-only today | #36 |
| **Parameter sweep runner** | Compare ε vs detection trade-offs | #30 |
| **Docker environment** | Reproducible 250K runs | #34 |
| **Interactive dashboard** | CSV/PNG export only | New issue |
| **Coverage fail threshold in CI** | Reported but not enforced | New issue |

---

## Engineering

| Feature | Notes | Suggested issue |
|---------|-------|-----------------|
| **Ruff in CI** | Local ruff configured; CI may only run pytest | #31 |
| **Type checking (mypy/pyright)** | Hints present, no CI gate | #37 |
| **CONTRIBUTING + CHANGELOG** | Onboarding docs | #28 |

---

## Filing a feature request

```markdown
## Type
**Feature**

## Summary
What capability and why it matters for the testbed.

## Scope
Minimal viable implementation.

## Acceptance criteria
- [ ] ...
- [ ] Tests or benchmark demonstrating value
```

Use label `enhancement`. Mark `good first issue` only if well-scoped and ≤ ~1 day of agent work.

---

## Out of scope (unless explicitly requested)

- Production deployment / HIPAA compliance certification
- Real wearable hardware integration
- Blockchain or cryptocurrency routing
- UI/mobile app for citizen agents
