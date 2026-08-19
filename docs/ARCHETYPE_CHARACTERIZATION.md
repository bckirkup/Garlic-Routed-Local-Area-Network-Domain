# Town archetype characterization (measured, not asserted)

Five committed scenarios (`examples/town_*.yaml`), each run once at its committed
seed for the full 1152 steps (4 days) with `PYTHONHASHSEED=0`, rectangular
backend, `k_min = 50`. All statistics below exclude the 288-step world-settling
prefix. Harness: `scripts/characterize_archetypes.py`; raw output
`docs/archetype_characterization.json`. No `src/` behaviour was changed to obtain
these numbers; the only instrumentation is a wrapper around `grid.dilated_zone`,
which is called once per issued broadcast.

## What was measured

| | college | tourist | mill | retirement | exurb |
|---|---:|---:|---:|---:|---:|
| agents | 3000 | 2500 | 1800 | 1500 | 2000 |
| wearables | 660 | 450 | 253 | 526 | 401 |
| agents / km² | 187.5 | 138.9 | 72.0 | 60.0 | 13.9 |
| **wearables / km²** | 41.3 | 25.0 | 10.1 | 21.0 | **2.8** |
| occupied wearable cells (mean/step) | 23.1 | 69.5 | 35.9 | 46.0 | 55.0 |
| **wearables per occupied cell (mean)** | **28.5** | 6.5 | 7.1 | 11.4 | 7.3 |
| dilated cells per broadcast (median) | 1 | 1 | 1 | 1 | 1 |
| dilated cells per broadcast (mean) | 1.15 | 2.19 | 1.63 | 2.88 | 2.47 |
| dilated zone area (mean, km²) | 0.046 | 0.088 | 0.065 | 0.115 | 0.099 |
| **wearables inside dilated zone (mean)** | 73.1 | 30.7 | **19.5** | 51.6 | 28.9 |
| broadcasts whose zone holds ≥ 50 wearables | 61% | 16% | **2%** | 47% | 5% |
| broadcasts / occupied zone / day | 16.4 | 3.3 | 2.0 | 4.4 | 2.6 |
| occupied zones alarming | 56% | 43% | 38% | 42% | 39% |
| cold-baseline wearable-step fraction | 0.000 | 0.00036 | 0.000 | 0.000 | 0.000 |
| benign instances overlapping a broadcast (mean) | 0.12 | 0.35 | 0.06 | 0.15 | **0.00** |
| broadcasts with no overlapping benign instance | 88% | 66% | 94% | 85% | **100%** |
| unexplained detection rate | 0.86 | 0.56 | 0.97 | 0.92 | 0.999 |
| ε / agent / day | 0.051 | 0.023 | 0.016 | 0.069 | 0.027 |

## Four things the numbers say, two of which contradict what I claimed

### 1. k-anonymity is dilated against residents, but only devices answer

`dilated_zone` grows the zone until the **resident** population reaches
`k_min`, and residents include the ~65–86% of agents with no wearable. The set
that can actually respond to a broadcast is the wearable set, and that is the
set an adversary observing responses gets to intersect. Measured against
wearables, the achieved anonymity set is 73 (college), 52 (retirement), 31
(tourist), 29 (exurb) and **19 (mill)** — so in the mill town 98% of broadcasts
are dilated over fewer than 50 respondents while the code's own criterion reads
as satisfied. The median dilation is **one cell in every archetype**: the
trigger cell alone clears the resident test, so dilation usually does nothing
at all.

This is not a scenario defect; it is a protocol-level accounting choice that the
archetypes made visible. Whether `k_min` should count respondents rather than
residents is a decision for you — it changes every published ε and every zone
footprint.

### 2. Sparse *area* does not produce sparse *cells* — the exurb prediction was wrong

I predicted the exurb would be the dilution-limited regime because it is 15×
sparser in devices per km². It isn't: schedule mobility funnels agents into
venues, so cell occupancy is set by venue structure, not by areal density. The
exurb averages 7.3 wearables per occupied cell — indistinguishable from the mill
town's 7.1 at 3.6× the areal density — and dilates to only 2.5 cells. The worst
anonymity case is the **mill town**, because low *adoption* (0.14) starves the
respondent set regardless of geography. Density per km² is the wrong knob for
this regime; wearable fraction is the right one.

### 3. The tourist town's defining regime does not exist yet

Cold-baseline wearable-steps: **0.036%** — four devices per day against 450.
Recurring adoption cohorts (`cohort_size: 4`, `interval_steps: 288`) are three
orders of magnitude too small to make cold-baseline inflow a regime, and every
other archetype measures exactly zero. The scenario documents that visitors are
approximated rather than modeled; the measurement says the approximation is
currently inert. Either the cohort schedule needs to move a substantial fraction
of the fleet each week, or turnover needs real arrival/departure.

### 4. The towns are not cluttered — and the clutter that exists is unscorable

85–100% of broadcasts have **no** registered benign instance overlapping them,
and multi-explanation overlap is ≤1.3% everywhere (0.0% in the exurb). Yet
detections *are* being explained (43% in the tourist town, 11% in the college
town) — because per-agent causes like exercise and sleep disruption reach every
wearable, while the benign *instance* registry only holds zone-local sources
(crowding, ILI, onboarding). So the explanatory load sits almost entirely in
causes the ask scorer cannot see: ask well-foundedness is scored against the
registry, and the registry is empty over the zones that trigger.

That is the concrete argument for the recurrence work: the archetypes vary
adoption, susceptibility and geography, but with no rhythm events running the
background has no stationary clutter level, which is exactly what the catalogue
predicted.

## What the archetypes do deliver

The college town is a genuinely distinct regime — 660 wearables concentrated
into 23 occupied cells, 16.4 broadcasts per occupied zone per day, 56% of
occupied zones alarming at once. It is the saturation case, though saturation by
*alarms*, not by benign explanation. Retirement is the highest-ε town
(0.069/agent/day) at the second-largest anonymity set. Mill is the floor case on
every device-side axis. Those three orderings are real and now measured.

## Not measured

No hazard is seeded in any archetype, so nothing here speaks to detection
latency, sensitivity, or the vulnerability-stratified questions the exposure
layer was built for. Single seed per archetype: the orderings are large enough
to be robust, but no dispersion is reported. Rectangular backend only; H3 was
not run.
