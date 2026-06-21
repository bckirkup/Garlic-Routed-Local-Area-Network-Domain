# GARLAND Feature Backlog

Items **not yet implemented** or optional stretch goals. Many former backlog items (#29–#38) are now on `main` — see `CHANGELOG.md` and `README.md`.

```bash
gh issue list -R bckirkup/Garlic-Routed-Local-Area-Network-Domain --state open --label enhancement
```

---

## Likely still open / stretch

| Area | Idea | Notes |
|------|------|-------|
| **Docker** | Reproducible container image | #34 if still open |
| **Dashboard** | Interactive results UI | CSV/PNG only today |
| **Formal privacy audit** | Prove DP claims | Design goals in README only |
| **Real homomorphic encryption** | Paillier/BFV tokens | Plaintext simulation today |
| **Mesh routing layer** | "Garlic-routed" network | Broadcast-and-filter only |
| **Coverage CI gate** | Fail below threshold | Coverage reported, not enforced |
| **Epidemiological calibration** | Fit SEIR to real outbreaks | Research extension |
| **Structured venues** | Schools, hospitals as graph | Beyond neighborhood clusters |
| **Wearable dropout** | Sensor failure model | All wearables always on |
| **Baseline warm-up** | Suppress early false anomalies | Optional config |

---

## Implemented on main (do not re-file)

- H3 + rectangular spatial backends
- Agent mobility (random walk)
- YAML/TOML config + `examples/`
- Parameter sweeps (`garland sweep`)
- Multi-plume / multi-outbreak support
- Optional NeuroKit2 synthesis + OpenWearables export
- Ruff + mypy CI
- CONTRIBUTING + CHANGELOG

---

## Filing new work

```markdown
## Type
**Feature** | **Bug fix** | **Enhancement** | **Documentation**

## Summary
...

## Acceptance criteria
- [ ] Tests or docs
- [ ] `pytest tests/ -v` and `ruff check src tests`
```

See `CONTRIBUTING.md` for branch naming and PR expectations.
