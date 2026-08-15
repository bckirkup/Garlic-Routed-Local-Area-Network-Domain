# Event catalogue and warrant classes

GARLAND reports what a detection warrants, rather than treating every
non-target detection as a false positive. The additive warrant taxonomy is:

| Class | Meaning |
| --- | --- |
| `TARGET` | The seeded disease or toxin hazard under test. |
| `ACTIONABLE_NON_TARGET` | A real harmful exposure that is not the target, such as heat strain or background ILI. |
| `EXPLAINED` | A real, harmless cause such as exercise, venue crowding, onboarding, or sleep disruption. |
| `ARTIFACT` | A measurement fault with no overlapping benign event, currently represented by sensor artifacts. |
| `UNEXPLAINED` | No known hazard or benign event overlaps the detection. |

The precedence is `TARGET`, `ACTIONABLE_NON_TARGET`, `EXPLAINED`, `ARTIFACT`,
then `UNEXPLAINED`. Existing hazard confusion-matrix, attribution, and
discrimination metrics retain their historical meanings. Warrant counters and
artifact/unexplained rates are additive reporting.

## Exposure layer

The confounder engine generates seeded per-agent evaluation attributes for
elderly status, air conditioning, outdoor work, and endurance athletics. It
also derives a bounded continuous heat-island factor from the agent's initial
position. These values are model-side evaluation ground truth: confounders may
use them to shape exposure and metrics may report them, but they never enter
`EncryptedToken`, `BroadcastQuery`, `PerturbedResponse`, or
`DisambiguationQuery`. They are not available to aggregation, spatial
dilation, classification, trigger gates, or ask content. The disambiguation
vocabulary therefore cannot probe age, housing, work, income, or other
exposure attributes.

The implemented heat advisory is a stable episode with a diurnal profile,
exposure-weighted affected agents, an air-conditioning reduction, a
night-time no-air-conditioning floor, and per-agent amplitude jitter. Its
active instance is not global: its affected-agent set is recomputed each step.
This makes heat exposure a spatially and socially structured actionable
non-target rather than a uniform city-wide offset.

Sleep disruption retains its in-engine draw but jitters the delay before onset
by a configurable number of steps, preventing an otherwise synchronized
06:00 wave.

## Ask scoring

An ask is well-founded when any active benign instance overlapping its zone has
the expected cause. Overlapping instances no longer allow a larger unrelated
instance to mask a matching cause. No benign truth enters the query payload or
trigger decision.

## Roadmap and assumptions

The catalogue is intentionally broader than this first implementation:

1. Exposure, warrants, and jitter are the foundation.
2. Cause-specific signature shape is the next needed step; current
   perturbations still share a common four-channel geometry.
3. Contact-based ILI, source events without onset, and compounding events remain
   future work.
4. Block fire and stadium/civic-victory events are deferred.
5. Subway fire, moving/non-surface footprints, and position loss require a
   separate spatial-model pass.

The assumption notes remain important: heat previously assumed uniform
exposure; sleep disruption assumed a fixed synchronized delay; ILI is still
household-seeded rather than contact-transmissive; and a persistent standing
condition can be absorbed by a personal baseline rather than detected as an
event. Community grief remains outside the ask vocabulary because it is not an
appropriate question for this system.
