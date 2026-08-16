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
The default night floor is `0.35`, calibrated so the overnight material
footprint remains non-empty without counting merely uncooled agents: at the
default `heat_wave_materiality_floor` of `0.5`, the overnight material set
contains vulnerable, non-AC agents while cooling remains excluded. The
materiality floor is an evaluation-side nominal-exposure threshold. It
determines which agents count as materially affected for provenance, warrant
attribution, and instance overlap only; it does not suppress or alter the
proportional perturbation applied to any agent with nonzero heat weight. This
keeps the physics graded while preventing residual AC exposure from making
the evaluation instance city-wide.

The implemented block fire is a fixed point-source smoke event in metre
coordinates. It applies a Gaussian distance falloff through three source
radii, recomputes exposure from current positions every step, and uses
elderly status only as an evaluation-side susceptibility factor. Physical
perturbations are proportional to the nonzero smoke weight, while the
`block_fire_materiality_floor` selects the materially exposed agents recorded
in the non-global `block_fire_0` instance. Its signature is respiratory-rate
dominant, with elevated heart rate, reduced HRV, positive respiratory content,
and no fever. The model has no responder-agent population, so responder
prioritization is not represented. Fire intensity is constant while active;
there is no ramp/decay envelope. The centre uses metre world coordinates and
must be set explicitly; the default is the world origin/corner.

The implemented stadium/civic victory is a synchronized, fan-only
sleep-disruption wave. A dedicated evaluation-only `sports_fan` attribute
selects the cohort, participation selects the fans who stay awake, and a
small per-agent onset jitter is followed by a linear decay. The non-global
`victory_0` instance contains participating fans currently receiving the
perturbation. Physiologically it is indistinguishable from individual sleep
disruption; only the model-side event registry distinguishes the victory
event. Fan membership and participation remain evaluation-only and are not
available to the protocol or ask vocabulary.

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
2. Contact-based ILI, source events without onset, and compounding events remain
   future work.
3. Subway fire, moving/non-surface footprints, and position loss require a
   separate spatial-model pass.

The assumption notes remain important: heat previously assumed uniform
exposure; sleep disruption assumed a fixed synchronized delay; ILI is still
household-seeded rather than contact-transmissive; and a persistent standing
condition can be absorbed by a personal baseline rather than detected as an
event. Community grief remains outside the ask vocabulary because it is not an
appropriate question for this system.
