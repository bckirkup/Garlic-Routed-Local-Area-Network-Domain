# Host phenotypes

GARLAND can optionally model host differences separately from device ownership.
The `hosts.enabled` switch defaults to `false`; disabled runs retain the
historical host and random-number paths.

## Phenotypes

- **Diabetic hosts** are sampled with age-weighted prevalence (10% by default).
  Their susceptibility is multiplied by 1.4. The testbed presents infection
  with a blunted inflammatory and temperature response, increased heart-rate
  response, and increased hypovolemia. Their symptomatic ramp is 1.3 times
  slower.
- **Frail elderly hosts** are the elderly demographic band when demographics are
  enabled, or an independent draw using the confounder elderly fraction
  otherwise. Susceptibility is multiplied by 1.6. The testbed reduces
  inflammatory and temperature presentation while increasing activity
  withdrawal, neuromotor fatigue, sleep disturbance, and pulmonary involvement.
- **Law-enforcement hosts** are sampled from adults and older adults. They are
  aligned with outdoor-worker confounding and receive a fourfold sleep-disruption
  probability to represent shift-work exposure.
- **Assistive-need hosts** are sampled from adults and older adults with an
  elderly-skewed weight. They are used for device-need gating.

Susceptibility multipliers compose multiplicatively and are capped at 3.0.
Hearables are available only to frail elderly, law enforcement, or assistive
need hosts; age affinity then favors elderly (2.0), older adults (1.2), and
adults (1.0).

These are testbed calibration choices, not clinical prevalence estimates.
Law-enforcement mobility toward incidents is not modeled yet. Diabetic
modeling is presentation-level; a CGM or glycemic channel is deferred to a
future PR.
