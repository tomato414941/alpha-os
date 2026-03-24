# Portfolio Runtime Principles

Design note. This document captures the current discussion around what the
hypotheses-first runtime should optimize for after the mainline is working.
It complements [RECOVERY.md](../RECOVERY.md) and [DESIGN.md](../DESIGN.md).

## Scope

This note is not a runbook and not the current operating truth. It is a
statement of design intent for the next stage of the hypotheses-first runtime.

This file is the right place for:

- portfolio / allocation terminology used by the current mainline
- near-term hypotheses-first design decisions
- current bottlenecks between research, observation, allocation, and execution

This file is not the right place for:

- full greenfield architecture
- migration-era archival detail
- operational runbooks

For those, prefer:

- [RECOVERY.md](../RECOVERY.md) for current operating truth
- [DESIGN.md](../DESIGN.md) for greenfield baseline and long-horizon architecture

The current trusted mainline remains:

1. `hypothesis-seeder`
2. `sync-signal-cache`
3. `produce-predictions`
4. `trade --once --venue paper`
5. `runtime-status`

## Current Mainline Summary

The current runtime is hypotheses-first:

- the source of truth for tradable ideas is `hypotheses.db`
- the live set is `status='active' AND stake>0`
- predictions are produced from live hypotheses into the prediction store
- `Trader` consumes those predictions and runs a bounded single-asset cycle
- readiness tracks whether this runtime can be operated repeatedly and safely

This means the next problem is no longer "how do we route legacy registry data
through the system". The next problem is "how do we turn a working runtime into
a better portfolio process".

## What Profitability Depends On

Profitability is not determined by a single component.

At minimum it depends on four distinct layers:

1. objective quality
   - the system must reward the right thing when it evaluates hypotheses
2. search power
   - the system must be able to explore enough of the hypothesis space
3. data quality
   - the system must observe enough reliable signal to support those evaluations
4. portfolio conversion
   - the system must convert good hypotheses into a good portfolio rather than a
     noisy equal-weight basket

This implies:

- a better evaluation metric is necessary but not sufficient
- more compute and more data help, but they do not remove the need for good
  portfolio construction
- a good research stack can still produce poor live results if weighting,
  correlation control, or cost handling are weak

## Desired Layering

The runtime should separate concerns more clearly.

### Hypothesis Layer

A hypothesis should only answer:

- what direction it predicts
- how strong that prediction is

It should not decide:

- whether it is trusted
- how much capital it receives
- how execution should happen

### Evaluation Layer

Evaluation should separate three different ideas:

- `research_quality`
  - why the hypothesis entered the live set in the first place
- `live_quality`
  - whether the hypothesis is still working now
- `estimated_quality`
  - the current quality estimate after combining research and live evidence

Research fitness is an entry signal. Live forward performance is an update
signal. `estimated_quality` is the current belief about the hypothesis. These
should not collapse into one field.

### Allocation Layer

Allocation should be a separate concern from evaluation.

The runtime should maintain a distinct concept:

- `allocation_trust`
  - how much capital this hypothesis deserves today if it is still eligible

The current codebase still uses the field name `stake` for storage and runtime
weighting. In this document, the intended meaning of that field is
`allocation_trust`, not "how good the hypothesis is".

The desired control loop is:

1. hypotheses emit predictions
2. the portfolio consumes those predictions
3. realized outcomes are recorded
4. lifecycle updates `status`
5. allocation updates `allocation_trust`
6. the next cycle allocates capital differently

This is the key loop that turns a hypothesis engine into an adaptive portfolio
system.

### Execution Layer

Execution should remain downstream from prediction and allocation.

It should be responsible for:

- deadbands
- minimum notionals
- cost and slippage handling
- inventory / position transitions

This separation matters because a poor trade outcome may come from execution
rather than signal quality.

## What "More Good Hypotheses" Means

The system should distinguish between:

- many candidate hypotheses
- many active hypotheses
- many effective independent bets

These are not the same thing.

### Candidate Count

More candidates are generally good because they increase search breadth.

### Active Count

More active hypotheses are not automatically better. Weak or redundant
hypotheses dilute the live set and make weighting harder.

### Effective Breadth

The thing that actually matters is effective breadth:

- how many independent positive-expectation bets exist after correlation,
  regime sensitivity, and cost are considered

In a fully idealized world, more truly good and sufficiently independent
hypotheses would always help. In practice:

- measured quality is noisy
- correlations rise under stress
- similar hypotheses often collapse into the same bet
- turnover and execution costs penalize many "good" ideas

Therefore the design goal is:

- broad search upstream
- constrained live set downstream
- concentrated capital on the most trusted independent bets

## Implications For `alpha-os`

The next stage should not focus on generating more hypothesis types first.
It should focus on improving how the current live set is trusted and weighted.

Priority order:

1. make `stake` respond to live forward evidence
2. keep selection and weighting on compatible scoring logic
3. add cost awareness to selection, not just execution
4. improve prediction calibration with rolling history rather than ad hoc tuning
5. improve attribution so the system can tell which hypotheses helped or hurt

In other words:

- the current bottleneck is not "we cannot produce predictions"
- the current bottleneck is "we are not yet converting hypothesis quality into
  capital allocation cleanly enough"

## Working Principles

Until the runtime is more mature, changes should follow these principles:

- optimize for removing bad hypotheses faster, not just adding new ones
- treat `stake` as a dynamic trust variable
- prefer effective breadth over raw hypothesis count
- keep selection, weighting, and cost handling logically aligned
- preserve bounded `oneshot` execution even as the portfolio logic gets richer

## Non-Goals For Now

The following are explicitly not the first priority:

- adding more legacy-compatible paths
- maximizing the number of live hypotheses
- building a permanent always-on runtime for this medium-frequency workflow
- hand-tuning individual indicators as the primary optimization method

## Open Design Questions

The next concrete design work should answer these questions:

1. How should `allocation_trust` be updated from forward performance?
2. How should `estimated_quality` and `allocation_trust` differ conceptually and numerically?
3. Should selection and weighting use the same score, or separate scores?
4. How should turnover and modeled cost penalize otherwise attractive hypotheses?
5. What should count as a distinct independent bet for portfolio construction?

## Concrete Proposal

This section proposes a concrete next-stage design for the current
hypotheses-first runtime. It is still a design note, but it is specific enough
to implement incrementally.

### Core Definitions

Use five separate quantities per hypothesis:

1. `research_quality`
   - historical OOS quality at registration or adoption time
   - slow moving
2. `live_quality`
   - rolling forward quality from realized live outcomes
   - fast moving
3. `observation_confidence`
   - how much live evidence is currently available
   - should not be confused with trust
4. `estimated_quality`
   - the current quality estimate after combining research and live evidence
5. `allocation_trust`
   - current capital allocation trust for an eligible hypothesis
   - stored today in the `stake` field

These quantities should not be collapsed into one field.

### Proposed Meanings

`estimated_quality` answers:

- is this hypothesis good?

`allocation_trust` answers:

- how much capital should this hypothesis receive today?

This distinction matters because:

- a promising but under-observed hypothesis can have reasonable quality but
  modest allocation trust
- a once-good but now-degrading hypothesis can retain historical quality while
  losing allocation trust quickly

This is the intended long-term interpretation:

- `research_quality` is a registration-time evaluation
- `live_quality` is an observed runtime evaluation
- `estimated_quality` is the current belief about the hypothesis
- `status` is an operational decision about whether it remains eligible
- `allocation_trust` is the runtime's current capital allocation decision

In other words, `stake` in the current code is not the same thing as "how good
the hypothesis is". It is the persisted storage field for
`allocation_trust`.

### Daily Control Loop

The desired daily loop is:

1. produce predictions from live hypotheses
2. trade the portfolio using current stakes
3. record realized forward contribution per hypothesis
4. recompute `live_quality`
5. recompute `estimated_quality`
6. update `status` if needed
7. update `allocation_trust`

This should happen once per bounded cycle. It should not depend on a daemon
that mutates state continuously.

## Proposed Scoring Model

### Quality

Keep the current blend shape in spirit:

- `estimated_quality = (1 - observation_confidence) * research_quality + observation_confidence * live_quality`

but tighten the meaning of `live_quality`.

`live_quality` should be computed on net forward outcomes:

- use realized forward contribution after modeled cost
- use a rolling window
- use the same portfolio objective family as the runtime where possible

That means:

- if the portfolio objective is log-growth, live quality should not be based on
  raw Sharpe by default
- if turnover is high, live quality should fall even if gross directional calls
  are good

### Early-Stage Live Quality

Small-sample live quality should not be treated as decision-quality evidence.

In particular, the runtime should not allow:

- 1 to 5 observations
- annualized metrics such as Sharpe or log-growth
- and very small realized volatility

to produce extreme `live_quality` values that leak into selection or
allocation.

The quality layer should therefore distinguish:

- `raw_live_quality`
  - the direct rolling metric computed from available forward returns
  - useful for monitoring and diagnostics
- `effective_live_quality`
  - the bounded or shrunk version that is actually allowed to affect
    `estimated_quality`

`estimated_quality` should be built from `effective_live_quality`, not from the
raw annualized metric directly.

### Recommended Early-Stage Rule

For the first safe implementation:

- compute `raw_live_quality` as today
- clip it to a bounded interval before use
- shrink it by observation count before it enters `estimated_quality`

Conceptually:

- `effective_live_quality = early_stage_scale * clipped_raw_live_quality`

Where:

- `clipped_raw_live_quality` prevents pathological small-sample extremes
- `early_stage_scale` rises from `0` toward `1` as observations accumulate

This is intentionally conservative. It means:

- early live observations remain visible
- but they do not dominate `estimated_quality`
- and they cannot overwhelm bootstrap-trust handoff through a single
  exaggerated annualized metric

### Responsibility

This is a quality-layer responsibility, not an allocation-layer patch.

The allocation layer should consume:

- `research_quality`
- `estimated_quality`
- `observation_confidence`

It should not be responsible for correcting pathological small-sample
annualization. That belongs in the construction of `live_quality` itself.

### Observation Confidence

`observation_confidence` should grow with the number of forward observations,
but it should not jump from zero to full trust abruptly.

Desired shape:

- low observations: mostly trust research quality
- medium observations: trust the blend
- high observations: trust the live track record

The current linear ramp is acceptable as a starting point.

## Proposed Allocation Trust Update

`allocation_trust` should be updated from current quality and recent
contribution, not from raw contribution mean alone.

### Why

A pure contribution mean is too fragile:

- one large move can dominate it
- it ignores observation confidence
- it does not distinguish "promising but uncertain" from "small but robust"

### Proposed Inputs

For each live hypothesis, compute:

- `estimated_quality`
- `observation_confidence`
- `recent_contribution_mean`
- `recent_contribution_vol`
- `turnover_penalty`
- `correlation_penalty`

### Proposed Target Allocation Trust

The target allocation trust should be a positive trust score rather than a raw
mean PnL.

Conceptually:

`target_allocation_trust = positive_part( estimated_quality - cost_penalty - crowding_penalty ) * confidence_scale`

Where:

- `cost_penalty` grows with turnover and modeled slippage
- `crowding_penalty` grows when the hypothesis is too similar to already trusted
  hypotheses
- `confidence_scale` increases with the amount of live evidence

The update itself should be smoothed:

- `next_allocation_trust = (1 - update_rate) * current_allocation_trust + update_rate * target_allocation_trust`

This avoids violent day-to-day capital jumps.

### Allocation Trust Constraints

The runtime should enforce:

- floor at zero
- per-hypothesis max allocation trust
- total normalization before portfolio weighting

This keeps allocation trust interpretable as trust mass rather than arbitrary
magnitude.

## Proposed Selection And Weighting Split

Selection and weighting should be related but not identical.

### Selection Score

Use selection to answer:

- which hypotheses are allowed into today's candidate set?

Selection should rank by:

- blended quality
- minimum observation threshold
- cost penalty
- correlation / crowding penalty

Selection is where redundancy should be removed aggressively.

### Weighting Score

Use weighting to answer:

- how much of the portfolio should each selected hypothesis control?

Weighting should be mostly driven by:

- normalized allocation trust

This means:

- quality determines whether the hypothesis survives selection
- allocation trust determines how much capital it gets once selected

This is cleaner than using one number for both jobs.

## Proposed Status Policy

`status` should be a coarse operational state, not a fine-grained score.

### `active`

Hypothesis is eligible for live allocation.

Requirements:

- structurally valid
- minimum trusted observations reached, or explicit bootstrap allowance
- non-zero allocation trust

### `paused`

Hypothesis is temporarily removed from allocation but may return.

Use for:

- short-term degradation
- missing data dependencies
- temporary operational issues

### `archived`

Hypothesis is retired.

Use for:

- structural invalidity
- repeated degradation over a long window
- permanently redundant or dominated hypotheses

## Proposed Cost Handling

Cost should appear before execution.

Execution must still model realized cost, but the selection and allocation
process should also penalize hypotheses that can only look good before cost.

At minimum:

- estimate turnover from recent signal path
- estimate modeled slippage from current execution model
- subtract a cost penalty in selection and stake update

This prevents the runtime from repeatedly selecting fragile high-turnover ideas.

## Proposed Correlation Handling

Correlation should be treated as a portfolio construction constraint, not only
as a post-hoc diagnostic.

The current shortlist plus correlation filter is directionally correct, but the
next step is to make correlation affect trust before the final top-N cut.

Desired behavior:

- hypotheses can remain active even when correlated
- but their marginal capital share should fall as the cluster becomes crowded

That means correlation should affect:

- selection score
- target stake

not just the final selected count.

## Implementation Order

The safest implementation order is:

1. keep current `quality` estimation, but make lifecycle use it explicitly
2. replace raw contribution-mean `stake` updates with smoothed target stake
3. add cost penalty into selection
4. add correlation penalty into target stake
5. only then revisit prediction calibration and broader search space expansion

This order improves capital allocation before increasing model complexity.

## What Should Not Change

The following should remain stable while this is implemented:

- hypotheses-first source of truth
- bounded `oneshot` runtime chain
- explicit readiness tracking
- separation between prediction generation and execution

The design goal is not to rebuild the runtime again. The goal is to improve the
portfolio logic inside the now-working mainline.

## Internal Market Analogy

One useful way to think about the hypotheses-first runtime is as an internal
prediction market.

In that analogy:

- a hypothesis is a market participant
- a prediction is that participant's directional view
- `allocation_trust` is trust capital
- contribution is the participant's payout signal
- lifecycle is the capital reallocation rule
- the trader is the clearing engine that turns those views into one portfolio

This analogy is helpful because it encourages the right questions:

- who should receive more trust capital?
- which participants are redundant?
- which views improve the portfolio on the margin rather than simply repeating
  an existing bet?

## What We Want To Borrow

The runtime should deliberately borrow some ideas from prediction markets and
systems like Numerai.

### Trust Expressed As Capital

`allocation_trust` should behave like trust capital rather than a static
weight.

That means:

- good recent behavior earns more capital
- degraded behavior loses capital
- confidence matters, not only raw score

### Reward By Marginal Contribution

The correct reward signal is not "did this hypothesis look good in isolation?"
but:

- how much did this hypothesis improve the portfolio on the margin?

This is why contribution-based updates are directionally correct.

### Penalize Crowding

If several hypotheses express the same effective bet, they should compete for
the same trust budget rather than each being rewarded independently.

This is the internal-market version of crowding and price competition.

## What We Cannot Fully Reproduce

Even with a strong internal market design, some things cannot be obtained in
full.

### External Information Discovery

Real markets and open tournament systems are powerful because independent
participants bring knowledge that is outside the system designer's model space.

An internal hypothesis set cannot fully reproduce that.

Even with large search and diverse generators, the system still shares:

- one data platform
- one set of generators
- one evaluation family
- one allocator

So the system can simulate internal diversity, but it cannot create truly
external information inflow by itself.

### Independent Incentives

In real markets, participants risk their own capital under partially independent
beliefs and constraints.

In an internal runtime, all capital allocation rules are centrally designed.
That gives safety and consistency, but it also limits the range of behaviors
that can emerge.

### True Heterogeneity

The strongest collective systems benefit from heterogeneous participants:

- different datasets
- different horizons
- different constraints
- different models of the world

Internal hypotheses can approximate this, but there is a ceiling because they
ultimately live inside one designed environment.

### Institutional Discipline

Markets impose discipline through actual gain, loss, and exit.

An internal lifecycle can mimic this, but it remains a designed rule. If the
rule is weak, bad hypotheses remain in the system longer than they should.

## Design Consequence

This means the runtime should aim for:

- the benefits of an internal market
- without pretending it is a substitute for true external discovery

Concretely:

- use stake as trust capital
- use marginal contribution as a reward signal
- force correlated hypotheses to compete for trust
- preserve bounded, centrally controlled execution

But do not assume that this alone replaces:

- new datasets
- genuinely different model families
- new sources of outside information

## Practical Conclusion

The internal-market framing is valuable, but it changes what "improvement"
means.

The goal is not:

- "recreate a real market inside the runtime"

The goal is:

- "build a disciplined internal capital allocation mechanism that makes better
  use of the hypotheses we already generate"

This framing should guide future work on:

- stake update rules
- correlation penalties
- lifecycle policy
- effective breadth rather than raw hypothesis count

## Allocation Trust Update Rule

The current implementation can already:

- record per-hypothesis contribution
- estimate current quality from forward observations
- store and update `stake`

But the current allocation logic is still too primitive because it is close to:

- `stake = mean(recent contributions)`

This is a reasonable bootstrap, but it is not the desired long-term rule.

### Why The Current Rule Is Not Enough

A plain recent-mean contribution rule has four problems:

1. it is too sensitive to one or two large outcomes
2. it ignores confidence in the live estimate
3. it ignores modeled cost
4. it does not make correlated hypotheses compete early enough

This means the next rule should be based on target trust, not raw recent PnL.

### Proposed Inputs

For each hypothesis `i`, define:

- `RQ_i`
  - research quality
- `LQ_i`
  - live quality
- `OC_i`
  - observation confidence
- `EQ_i`
  - estimated quality
- `M_i`
  - recent mean marginal contribution
- `V_i`
  - recent contribution volatility
- `K_i`
  - modeled cost penalty
- `R_i`
  - crowding / correlation penalty

These do not all need to be perfect immediately. The order of approximation can
be:

1. `EQ_i`, `OC_i`, `M_i`
2. then `K_i`
3. then `R_i`

### Proposed Trust Score

The next internal quantity should be a trust score:

`T_i = positive_part( w_q * EQ_i + w_m * M_i - w_k * K_i - w_r * R_i )`

Interpretation:

- `EQ_i` says whether the hypothesis deserves trust
- `M_i` says whether it has recently helped on the margin
- `K_i` says whether cost is eating the edge
- `R_i` says whether the bet is already crowded

This trust score is not yet the portfolio weight. It is only the target
allocation-trust driver.

### Observation Scaling

Trust should scale with evidence.

Define:

`A_i = OC_i * T_i`

This produces the intuition we want:

- good but under-observed hypotheses get some trust, but not full trust
- well-observed and still-good hypotheses get more trust
- degraded hypotheses lose trust even if they once had a good prior

This is also where the current design question lives. If `OC_i` only measures
live observations, then hypotheses with strong `research_quality` but zero live
history can be assigned too little trust. A later revision may need an explicit
bootstrap trust term rather than relying only on `OC_i`.

## Bootstrap Trust

The runtime should not force a newly admitted hypothesis to earn all trust from
live observations alone.

That would collapse the following distinct ideas into one mechanism:

- whether the hypothesis was promising at admission time
- whether enough live evidence has been collected yet
- whether the runtime should allocate capital to it today

These should remain separate.

### Core Principle

`research_quality` should seed initial allocation trust, but it should not be
identical to allocation trust.

The runtime should treat bootstrap trust as:

- an initial allocation prior
- derived from admission-time or bootstrap-time research evidence
- gradually replaced by live evidence as observations accumulate

### Proposed Terms

Use three distinct quantities:

- `bootstrap_trust`
  - initial trust derived from `research_quality`
- `live_allocation_signal`
  - trust signal derived from live evidence
- `allocation_trust`
  - the combined current capital-allocation decision

### Intended Flow

At zero live observations:

- `allocation_trust` should be mostly bootstrap-driven

At medium live observations:

- `allocation_trust` should be a blend of bootstrap trust and live evidence

At high live observations:

- `allocation_trust` should be mostly live-driven

This implies a blend shape like:

`allocation_trust_i = (1 - OC_i) * bootstrap_trust_i + OC_i * live_allocation_signal_i`

Where:

- `OC_i` is `observation_confidence`
- `bootstrap_trust_i` comes from `research_quality`
- `live_allocation_signal_i` comes from runtime evidence

### Responsibility Split

- generation / bootstrap / admission
  - produces `research_quality`
- allocation layer
  - converts `research_quality` into `bootstrap_trust`
- quality layer
  - produces `live_quality` and `observation_confidence`
- allocation layer
  - converts live evidence into `live_allocation_signal`
  - blends bootstrap and live components into `allocation_trust`

This is important:

- `research_quality` is an evaluation output
- `bootstrap_trust` is an allocation input

They should not be stored or discussed as the same quantity.

### Design Consequence

The current phase-1 shape:

`target_allocation_trust = observation_confidence * base_trust`

is too strict for under-observed hypotheses because it implicitly assumes:

- zero live confidence means zero target allocation trust

That is too conservative for a hypotheses-first runtime that admits ideas
based on research evidence.

The next design step should therefore be:

- keep `estimated_quality` for selection and status decisions
- add explicit `bootstrap_trust` for initial allocation
- let `observation_confidence` control the handoff from bootstrap-driven trust
  to live-driven trust

### Practical Intent

The goal is not to give every new hypothesis large capital immediately.

The goal is:

- avoid collapsing all new trust to zero
- preserve a small but meaningful initial allocation for research-backed ideas
- transition toward live evidence without abrupt discontinuities

This makes the runtime:

- more faithful to the distinction between evaluation and allocation
- less dependent on immediate live observation volume
- easier to reason about when new hypotheses enter the active set

### Recommended First Approximation

The first implementation should keep bootstrap trust simple.

Use:

`bootstrap_trust_i = max( 0, bootstrap_weight * research_quality_i )`

Where:

- `bootstrap_weight` is a conservative scalar less than or equal to 1
- `research_quality_i` is already normalized into the same broad scale as
  `estimated_quality`

Then define:

`live_allocation_signal_i = positive_part( w_q * estimated_quality_i + w_m * M_i )`

And blend:

`target_allocation_trust_i = (1 - OC_i) * bootstrap_trust_i + OC_i * live_allocation_signal_i`

This gives a simple handoff:

- new hypotheses start from research-backed trust
- observed hypotheses transition toward live-driven trust
- no separate special-case path is needed after the blend is in place

### Why This Approximation Is Good Enough First

It avoids premature complexity:

- no separate regime logic yet
- no explicit cost penalty yet
- no explicit crowding penalty yet
- no nonlinear bootstrap schedule yet

But it still fixes the key conceptual problem:

- zero live observations no longer imply zero target allocation trust

### What To Avoid

The first bootstrap-trust implementation should not:

- set `allocation_trust = research_quality`
- skip smoothing
- assign equal trust to every active hypothesis
- hardcode a permanent minimum allocation independent of quality

The bootstrap term should be:

- quality-aware
- conservative
- temporary in influence as live evidence grows

### Initial Bootstrap Parameters

The first bootstrap-trust implementation should explicitly choose:

- a normalization rule for `research_quality`
- a conservative `bootstrap_weight`

#### Recommended `bootstrap_weight`

Start with:

- `bootstrap_weight = 0.25`

Meaning:

- bootstrap trust can seed allocation
- but it should not dominate the portfolio before live evidence exists

This is intentionally conservative. Bootstrap trust should be strong enough to
avoid collapsing every new idea to zero, but weak enough that the runtime still
needs live confirmation before allocating meaningfully larger capital.

#### Recommended Normalization

`research_quality` should be normalized into a bounded non-negative score
before it becomes bootstrap trust.

Use:

- `normalized_research_quality in [0, 1]`

Then:

`bootstrap_trust_i = bootstrap_weight * normalized_research_quality_i`

This makes bootstrap trust interpretable and prevents raw OOS metrics with
different units from leaking directly into allocation.

#### First Metric-Specific Approximation

For a first implementation, use simple clipping rules:

- if the runtime objective is `sharpe`:
  - `normalized_research_quality = clip( oos_sharpe / 2.0, 0, 1 )`
- if the runtime objective is `log_growth`:
  - `normalized_research_quality = clip( oos_log_growth / 0.20, 0, 1 )`

These constants are not meant to be universal truth. They are calibration
anchors that put bootstrap trust onto a stable first-pass scale.

Interpretation:

- Sharpe around `2.0` is already treated as a fully trusted bootstrap score
- annualized log growth around `0.20` is already treated as a fully trusted
  bootstrap score

Anything stronger than that is clipped rather than allowed to explode the
initial allocation.

#### Missing Research Quality

If a hypothesis has no `research_quality`, the first implementation should use:

- `normalized_research_quality = 0`
- therefore `bootstrap_trust = 0`

That keeps the rule clean:

- research-backed ideas get bootstrap trust
- non-research-backed ideas must earn trust from live evidence

If this later proves too harsh for specific generators, that should be handled
as an explicit admission or bootstrap policy, not by smuggling synthetic trust
into the allocator.

## Unscored Hypotheses

The runtime should explicitly distinguish between:

- research-backed hypotheses
- unscored exploratory hypotheses

This matters because a hypotheses-first system can easily generate many active
hypotheses that have never earned either:

- admission-time research evidence
- enough live evidence to justify allocation

Those hypotheses should not silently inherit the same runtime treatment as
research-backed hypotheses.

### Core Principle

If a hypothesis has:

- no `research_quality`
- and no meaningful live evidence

then it should not receive positive bootstrap trust.

This does not automatically mean it must be archived immediately. It means the
system should not pretend that it already deserves capital.

### Recommended Policy

For the current design, use the following rule of thumb:

#### Research-backed hypotheses

- may enter `active`
- may receive `bootstrap_trust`
- may compete for allocation immediately, but conservatively

#### Unscored exploratory hypotheses

- may exist in the store
- may remain `active` for exploration if the runtime intentionally wants that
- but should begin with:
  - `bootstrap_trust = 0`
  - `allocation_trust = 0` until live evidence exists

This keeps exploration and capital allocation separate.

### Design Consequence

The system should not rely on "active means capital-eligible" as a universal
rule.

The cleaner interpretation is:

- `status=active`
  - the hypothesis is structurally and operationally eligible to be observed
- `allocation_trust > 0`
  - the hypothesis is currently capital-eligible

That means unscored exploratory hypotheses may be:

- `active`
- but still outside the current capital allocation set

This is acceptable if it is intentional and visible.

### Current Runtime Constraint

The current implementation still uses:

- `status='active' AND stake>0`

as the effective live allocation set.

So, in the current code:

- `stake` continues to serve two jobs at once:
  - persisted allocation trust
- practical capital-eligibility switch

This is not the ideal long-term model, but it is the current implementation
constraint.

It also has a concrete consequence:

- hypotheses with `stake=0` are currently excluded not only from trading
  allocation
- but also from prediction production and lifecycle observation

So, under the current implementation, `stake=0` means:

- not capital-backed
- not observation-backed

This is too strong for exploratory hypotheses.

### First Implementation Rule

Given that constraint, the first implementation should use the simplest safe
rule:

- keep exploratory hypotheses `active` if needed for observation
- but register or update them with `stake = 0` unless they have:
  - `research_quality`, or
  - meaningful live evidence

This preserves the current runtime contract while moving the semantics in the
right direction.

However, this rule cannot be applied safely to exploratory random DSL
hypotheses until the runtime has a separate observation path.

Otherwise:

- unscored hypotheses would receive `stake=0`
- `stake=0` would remove them from prediction production
- no live evidence would ever be produced
- and they could never graduate out of the exploratory state

Therefore the actual safe sequencing is:

1. separate `observation-eligible` from `capital-eligible`
2. allow exploratory hypotheses to produce predictions and collect evidence
   without receiving capital
3. only then use `stake=0` systematically for unscored hypotheses

Until that split exists, the runtime must either:

- keep some exploratory hypotheses capital-eligible, or
- introduce a distinct non-capital observation path

This is the main implementation constraint behind the current design.

Concretely:

- `status`
  - remains the coarse operational state
- `stake`
  - remains the stored allocation-trust value
  - also controls immediate capital eligibility in the current runtime

That means the current runtime can evolve toward the cleaner model without
requiring an immediate schema split.

### Recommended Next Simplification

In the longer run, the runtime should probably make this explicit with a
separate concept such as:

- `observation-eligible`
- `capital-eligible`

But the first implementation does not need a schema rewrite. The simpler rule
is enough:

- only research-backed or live-proven hypotheses should receive allocation
  trust

### Practical Implication For Random DSL

Current random DSL generation is exploratory by default.

Therefore, until it gains one of the following:

- explicit `research_quality`
- meaningful live evidence

it should be treated as:

- searchable
- observable
- but not automatically capital-backed

This preserves the role of random search without letting it silently dominate
the active allocation pool.

### Smoothed Allocation Trust Update

The runtime should not jump directly to `A_i`.

Use:

`allocation_trust_i(next) = (1 - update_rate) * allocation_trust_i(now) + update_rate * A_i`

where `update_rate` is a slow update rate.

This gives:

- smoother capital movement
- less whipsaw from one-day noise
- more interpretable allocation-trust transitions

### Floors, Ceilings, And Normalization

After the smoothed update:

- clip allocation trust at zero
- optionally cap individual allocation trust
- normalize only at weighting time, not in storage

This preserves allocation trust as trust mass rather than forcing the database
to store weights that depend on the current peer set.

## Relationship Between Quality And Allocation Trust

The intended relationship is:

- estimated quality is a belief score
- allocation trust is a capital score

Estimated quality should answer:

- should this hypothesis remain credible?

Allocation trust should answer:

- how much allocation does this hypothesis deserve today?

This leads to an important asymmetry:

- quality may decay slowly
- allocation trust should be allowed to decay faster

That lets the system reduce capital exposure before it fully gives up on a
hypothesis.

## Ownership Map

The design should assign responsibility clearly.

- generation / adoption layer
  - creates `research_quality`
- quality layer
  - computes `live_quality`, `observation_confidence`, and `estimated_quality`
- lifecycle layer
  - updates `status` using quality and operational evidence
- allocation layer
  - updates `allocation_trust` and persists it in the `stake` field
- trader
  - consumes persisted `allocation_trust` during weighting and execution

This split is the main design principle. Evaluation and capital allocation are
related, but they are not the same responsibility.

This also means:

- research quality should influence initial trust
- live quality should influence ongoing trust
- but neither should be treated as identical to allocation trust itself

`stake` remains a downstream storage field for allocation trust.

## Status Transition Proposal

Status changes should be coarser and slower than allocation trust changes.

### Recommended rule of thumb

- if estimated quality is still acceptable but recent trust is weak:
  - keep `active`, reduce `allocation_trust`
- if quality is uncertain and observations are thin:
  - keep `active` or `paused` depending on operational confidence
- if quality degrades persistently and trust remains near zero:
  - move to `paused`
- if structural invalidity or long-run domination is clear:
  - move to `archived`

This preserves an important distinction:

- allocation trust is the first line of defense
- status is the stronger administrative action

## Selection Rule Proposal

Selection should occur in two stages.

### Stage 1: Eligibility

A hypothesis is eligible only if:

- `status == active`
- `allocation_trust > 0`
- observations are above a minimum floor, or it is still in a deliberate
  bootstrap grace period

### Stage 2: Candidate Ranking

Eligible hypotheses are ranked by a candidate score:

`candidate_score_i = EQ_i - K_i - R_i`

This score decides who reaches the shortlist.

It should not be replaced by allocation trust, because:

- allocation trust is path dependent
- candidate ranking should still notice a promising hypothesis whose trust has
  not yet fully caught up

## Weighting Rule Proposal

Once the shortlist is selected, weights should be driven primarily by
allocation trust.

Conceptually:

`portfolio_weight_i proportional to normalized(allocation_trust_i)`

This means:

- estimated quality drives entry and retention
- allocation trust drives capital share

If two hypotheses are both admitted, the one with stronger accumulated trust
gets more capital.

This is the key conceptual split:

- `estimated_quality` decides whether a hypothesis deserves consideration
- `allocation_trust` decides how much allocation an eligible hypothesis receives

## Cost Penalty Proposal

The first approximation of cost penalty should be simple and internal.

Possible proxy:

- recent signal turnover
- multiplied by modeled slippage and commission from the execution config

This is not perfect, but it is enough to stop obviously high-turnover
hypotheses from looking artificially attractive.

## Correlation Penalty Proposal

The first approximation of crowding penalty should be based on similarity to
already-trusted hypotheses.

Possible proxy:

- average absolute correlation to the top-trust cluster

That penalty should not force immediate archival. It should mainly:

- reduce target allocation trust
- reduce candidate score

This lets similar hypotheses survive as backup capacity while preventing them
from all receiving equal trust.

## Phased Implementation Plan

A practical sequence is:

1. replace contribution-mean allocation updates with smoothed trust-score updates
2. include estimated quality in the lifecycle update path
3. add simple turnover-based cost penalty
4. add simple correlation-based crowding penalty
5. revisit whether status transitions should become more aggressive

This sequence is intentionally incremental. It improves allocation without
requiring the whole runtime to be redesigned again.

## Phase 1 Minimal Implementation

The first implementation should intentionally be narrower than the full design.

### Included In Phase 1

- `estimated_quality`
- `observation_confidence`
- `marginal_contribution`
- `stake_update_rate`

### Deferred To Phase 2

- explicit `cost_penalty`
- explicit `crowding_penalty`
- more aggressive status automation
- richer early-stage live-quality shaping beyond simple clipping and shrinkage

This keeps the first implementation small enough to test without redesigning
selection and execution at the same time.

## Phase 1 Rule

For phase 1, define:

- `base_trust = positive_part( w_q * estimated_quality + w_m * marginal_contribution )`
- `target_allocation_trust = observation_confidence * base_trust`
- `next_allocation_trust = (1 - stake_update_rate) * current_allocation_trust + stake_update_rate * target_allocation_trust`

This is deliberately conservative because:

- it uses already-available quantities
- it preserves confidence scaling
- it avoids introducing speculative penalties too early

### Phase 1 Quality Guardrail

Phase 1 should also explicitly guard against small-sample live-quality spikes.

The first acceptable rule is:

- compute `raw_live_quality` from forward returns
- clip `raw_live_quality` into a bounded interval before blending
- multiply it by an observation-count scale before producing
  `effective_live_quality`
- use `effective_live_quality` inside `estimated_quality`

This keeps phase 1 simple while preventing 2-day or 3-day annualized metrics
from acting like mature live evidence.

## Phase 1 Parameter Intent

The first parameter set should favor stability.

Guiding intent:

- `w_q` should dominate `w_m`
  - trust should move primarily on estimated quality, not one-day contribution
- `stake_update_rate` should be small
  - allocation trust should move gradually rather than jump

Recommended qualitative defaults:

- quality-first
- contribution-second
- slow allocation-trust movement
- zero floor

The point of phase 1 is not to maximize responsiveness. It is to prove that
allocation trust can become a live trust signal without destabilizing the
current runtime.

## Phase 1 Initial Parameters

The following initial parameters are the recommended starting point.

### Proposed Defaults

- `quality_weight = 1.0`
- `marginal_contribution_weight = 0.25`
- `stake_update_rate = 0.10`
- `target_allocation_trust_floor = 0.0`

### Why These Defaults

`quality_weight = 1.0`

- estimated quality is the most stable signal currently available
- it should remain the primary trust anchor in phase 1

`marginal_contribution_weight = 0.25`

- recent contribution should matter
- but it should not dominate estimated quality yet
- this keeps one or two strong days from overwhelming the trust signal

`stake_update_rate = 0.10`

- this moves 10% of the way toward the new target each cycle
- it is conservative enough to avoid violent trust oscillation
- but still fast enough to react over the course of several daily cycles

`target_allocation_trust_floor = 0.0`

- phase 1 should not introduce synthetic minimum trust
- zero should remain the clean lower bound

### What This Means In Practice

With these defaults:

- allocation trust moves slowly
- quality matters more than short-term contribution
- hypotheses need repeated evidence to gain meaningfully more trust
- a degrading hypothesis loses capital gradually before any stronger status
  transition is considered

### Explicit Non-Defaults For Phase 1

The following should remain disabled or implicit in phase 1:

- no explicit cost penalty coefficient yet
- no explicit crowding penalty coefficient yet
- no automatic pause/archive based purely on one low target allocation trust update

Those mechanisms should be introduced only after the system proves stable under
the simpler trust rule.

## First Implementation Boundary

The first code change should be limited to the lifecycle path.

That means:

- update lifecycle math
- keep trader weighting logic mostly unchanged
- keep selection logic mostly unchanged
- observe how allocation trust evolves under the new rule

Only after that should the runtime use cost and crowding penalties in the
selection score itself.

## Research Quality Supply Gap

The hypotheses-first runtime still has an asymmetric `research_quality` path.

Current state:

- bootstrap technical and ML hypotheses enter with explicit
  `oos_sharpe` / `oos_log_growth`
- random DSL hypotheses generally enter without explicit registration-time
  research quality
- this means the runtime still relies on:
  - `research_backed` bootstrap seeds
  - `live_proven` promotions
  more than on a uniform research-quality pipeline

This is functional, but not the desired end state.

The desired end state is:

- every hypothesis that is eligible for live observation should have one of:
  - explicit `research_quality`
  - explicit `unscored/exploratory` status
- registration should make that distinction visible at insertion time
- lifecycle should not need to guess whether a zero-quality hypothesis was:
  - intentionally exploratory
  - or simply missing research metadata

In other words:

- `research_quality` should be a first-class registration output
- not an optional metadata path that only some seed families populate

## Before Profitability

The system should not treat short-term paper operation as proof of
profitability. Before the runtime can plausibly claim it is beginning to
produce real edge, the following work remains.

### 1. Uniform Registration Semantics

Needed:

- a clean distinction between:
  - `research_backed`
  - `exploratory`
  - `live_proven`
- a consistent `research_quality` supply path for hypotheses-first registration

Why it matters:

- without this, the runtime still depends too heavily on bootstrap seeds and
  late-stage promotion heuristics

### 2. Venue-Actionable Quality

Needed:

- quality estimates that reflect what the venue can actually monetize
- for example:
  - long-only venues should not reward bearish hypotheses in the same way as
    long-short venues

Why it matters:

- a hypothesis can be directionally correct but operationally unusable on a
  given venue

### 3. Observation Eligibility vs Capital Eligibility

Needed:

- a true separation between:
  - observation-active hypotheses
  - capital-backed hypotheses

Current limitation:

- `stake > 0` still acts as both:
  - allocation trust
  - capital eligibility switch

Why it matters:

- exploratory hypotheses should be observable without automatically receiving
  capital

### 4. Better Capital Set Diversity

Needed:

- cluster-aware control for both:
  - bootstrap seeds
  - promoted live hypotheses
- breadth logic that reflects actual portfolio bets rather than raw count

Why it matters:

- many hypotheses can still collapse into a few effective bets

### 5. Selection / Weighting / Venue Alignment

Needed:

- agreement between:
  - who is capital-backed
  - who is selected
  - who produces an actionable long signal

Current symptom:

- the system can have a non-trivial capital-backed set and still end in
  `no_delta` because the final combined view is bearish on a long-only venue

### 6. Cost and Turnover Awareness

Needed:

- explicit lifecycle and allocation penalties for:
  - turnover
  - modeled slippage
  - execution-unfriendly hypotheses

Why it matters:

- paper-correct hypotheses can still be negative after realistic trading costs

### 7. More Than One Asset and Regime

Needed:

- validation beyond a single BTC paper loop
- at minimum:
  - more than one asset
  - more than one market regime
  - more than one observation window

Why it matters:

- profitability claims from one asset and one short regime are not credible

### 8. Longer Forward Evidence

Needed:

- materially more live forward observations than the current short horizon

Why it matters:

- until the runtime accumulates more evidence, many choices remain dominated by
  uncertainty rather than demonstrated edge

## What To Do Next

The practical order should be:

1. continue bounded paper observation in the background
2. avoid BTC-specific or provisional-hypothesis-specific tuning
3. improve general structure only
4. prioritize:
   - uniform registration semantics
   - venue-actionable quality
   - observation/capital separation
5. defer aggressive optimization until more forward evidence exists

This keeps the project moving without pretending that current paper behavior is
already a profitability proof.

## Exploration Layer

The runtime should not treat all generated hypotheses as the same kind of
object.

The exploration stack should eventually separate at least three streams:

### 1. Bootstrap / Research-Backed Seeds

These are hypotheses that enter with explicit registration-time quality.

Examples:

- technical bootstrap hypotheses
- ML bootstrap hypotheses
- future manually-reviewed or batch-scored seeds

Role:

- provide stable anchors
- define a minimum research-backed capital set
- act as a baseline against which exploratory promotion can be measured

### 2. Exploratory Random Search

These are hypotheses created primarily to explore the search space.

Examples:

- random DSL proposals
- broad feature/operator sampling

Role:

- discover regions of the hypothesis space the system would not reach by hand
- create observation-only candidates
- provide raw material for later research scoring or live promotion

Constraints:

- they should not be assumed research-backed at registration time
- they should normally enter as observation-active, not capital-backed

### 3. Guided Search

This is the layer that should eventually become the main search engine.

Examples:

- mutation of promising hypotheses
- recombination of compatible hypotheses
- neighborhood search around live-proven or research-backed ideas
- cluster-aware search that prefers underrepresented bets

Role:

- improve on what already appears promising
- spend search budget near productive regions instead of remaining purely random
- raise the probability that new candidates are both distinct and useful

## Exploration Principle

Random search is acceptable as an entry mechanism, but it should not remain the
center of the system.

The desired balance is:

- keep random exploration alive as a source of novelty
- use research-backed seeds as stable anchors
- move the main search budget toward guided exploration over time

In other words:

- `random` should remain a source of idea generation
- not the final form of portfolio construction

## What This Means Right Now

In the current runtime, the correct near-term posture is:

- keep random DSL hypotheses as explicit `exploratory/unscored` registrations
- do not grant them capital by default
- let them earn promotion through either:
  - later research scoring
  - meaningful live evidence

The next design step after uniform registration semantics is therefore not
"more random generation". It is:

- define a separate research-scoring stage for exploratory hypotheses
- design guided search so that the system does not rely on pure randomness as
  its main engine

## Research Scoring Stage

The next hypotheses-first addition should be a separate `research scorer`
stage.

Its job is not to trade and not to allocate capital. Its job is to turn
explicitly exploratory hypotheses into explicitly scored hypotheses when enough
offline evidence exists.

### Why It Should Be Separate

The runtime should not force `hypothesis-seeder` to do research evaluation at
insert time.

Reasons:

- generation and evaluation have different cost profiles
- generation should remain cheap and broad
- research scoring should be batchable and restartable
- poor or partial offline scoring should not block observation registration

Therefore the clean split is:

1. `hypothesis-seeder`
   - generates and registers exploratory hypotheses
2. `research scorer`
   - evaluates selected exploratory hypotheses offline
   - writes `research_quality`
3. `lifecycle`
   - decides capital eligibility and allocation trust

### Inputs

The first version of the `research scorer` should consume:

- hypotheses from `hypotheses.db`
- only hypotheses that are:
  - `status='active'`
  - `source='random_dsl'` or another exploratory source
  - `research_quality_status='unscored'`
- cached data from `signal_cache.db`
- the same portfolio objective already used by the runtime

The first version should not depend on:

- live paper results
- current stake values
- capital-backed status

### Outputs

The `research scorer` should write only metadata updates, not status changes by
default.

Minimum outputs:

- `oos_sharpe`
- `oos_log_growth`
- `prior_quality_source='batch_research_score'`
- `research_quality_status='scored'`
- `research_scored_at`
- optional diagnostics such as:
  - `research_eval_window_days`
  - `research_eval_min_days`
  - `research_eval_result`

It should not automatically:

- set `stake`
- set `capital_backed`
- set `live_proven`
- change `status`

Those remain lifecycle responsibilities.

### Selection of What To Score

The first scorer should not evaluate every exploratory hypothesis blindly.

A reasonable first filter is:

- structurally valid
- not archived
- not already scored
- not obviously redundant by semantic key

Later versions may prioritize:

- hypotheses with enough observation history
- hypotheses that are semantically novel
- hypotheses that sit near promising clusters

### Failure Semantics

Failure to score is not the same as failure to observe.

The `research scorer` should distinguish:

- `scored`
- `failed_evaluation`
- `insufficient_history`
- `skipped_redundant`

That distinction should be visible in metadata, so lifecycle does not confuse:

- "not yet scored"
- with
- "scored and weak"

### First Implementation Boundary

The first implementation should be intentionally narrow.

It should:

- score exploratory DSL hypotheses only
- compute the same OOS metrics already used elsewhere in the codebase
- update hypothesis metadata in place
- leave `status` and `stake` untouched

The point of the first scorer is not to complete the full research pipeline.
It is to create a clean, hypotheses-first source of `research_quality` outside
of bootstrap seeds.

## If Rebuilding From Scratch

This section records the main architectural lessons from the current runtime.
The detailed greenfield baseline lives in [DESIGN.md](../DESIGN.md). The goal
here is narrower: capture what the current mainline has taught us about what
should and should not be coupled.

### What Would Stay The Same

These concepts still look correct:

- `hypotheses.db` as the source of truth
- explicit separation between:
  - `research_quality`
  - `live_quality`
  - `allocation_trust`
- exploratory hypotheses entering as observation-only first
- venue-aware quality rather than venue-agnostic scoring
- breadth measured as effective independent bets rather than raw count

In other words, the current design direction appears stronger than the current
code shape.

### What Would Be Different

A fresh implementation would likely avoid several transitional couplings:

- `stake` would not also serve as the capital-eligibility switch
- `observation-active` and `capital-backed` would be distinct persisted sets
- `lifecycle` would not own as much policy surface
- `Trader` would not own as much orchestration, selection, weighting, tracking,
  and reporting at once
- daily, hourly, and non-daily signal catalogs would be separated from the
  beginning
- bootstrap, batch-scored, and live-proven hypotheses would enter the capital
  path through a unified policy model rather than accumulated migrations

### Main Greenfield Differences

If rebuilt from zero, the biggest differences would be:

1. `stake` would start as pure persisted `allocation_trust`
   - it would not double as the capital-eligibility switch
2. observation and capital would be separate from day one
   - `observation-active`
   - `capital-eligible`
   - `capital-backed`
3. the runtime would start with thinner orchestration boundaries
   - CLI would only dispatch
   - scoring, backfill, and rebalance would live in services
4. research scoring would be a first-class stage from the start
   - not something bootstrap seeds have and exploratory hypotheses gain later
5. signal catalogs would be resolution-aware from the beginning
   - daily, hourly, and sparse event families would not share one implicit path

This is close to what the current code is converging toward, but it would have
been cleaner to start there than to migrate there.

### What Has Already Moved In That Direction

The current codebase has now taken a small but meaningful step toward that
decomposition:

- `HypothesisStore` exposes separate queries for:
  - `observation-active`
  - `capital-eligible`
  - `capital-backed`
- rebalance plan construction and cap application no longer live only in CLI
  orchestration; they have started moving into a dedicated service layer
- trading candidate ranking now prefers explicit capital eligibility rather
  than treating `stake > 0` as the only source of truth

This is still transitional:

- `stake` remains the persisted storage field for `allocation_trust`
- `stake > 0` still functions as the concrete backed/not-backed boundary in
  several places
- `capital_eligible` is still represented through lifecycle metadata rather
  than its own first-class persisted field

So the current state is not the clean final model yet, but it is no longer
purely implicit either.

### Why The Current Project Still Matters

The current project has still surfaced design truths that are worth preserving
in any rebuild:

- `live_proven` and `actionable_live` are not the same
- bearish-but-correct hypotheses should not be treated as directly actionable
  in a long-only venue
- `no_delta` is a normal outcome, not necessarily a runtime failure
- guided diversity matters more than simply increasing hypothesis count
- exploratory generation, research scoring, and capital allocation should be
  separate stages

So the main lesson is:

- the current implementation still contains migration-era seams
- but the underlying design vocabulary is now much better than it was at the
  start of the recovery
