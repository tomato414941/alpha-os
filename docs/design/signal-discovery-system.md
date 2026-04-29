# Signal Discovery System

## Purpose

This document defines the greenfield target for `alpha-os` without assuming
anything about the current implementation.

The intended system is not:

- a loop that runs one signal at a time
- a registry of asset-specific signals
- a collection of backtests glued to a portfolio policy

The intended system is:

- a large-scale observation consumer
- a representation engine
- a signal-discovery discovery and reduction engine
- a belief-to-decision engine

The goal is not to enumerate as many named signals as possible.

The goal is to discover, retain, and act on independent predictive structure.

## Signal Expression Language

If `alpha-os` introduces a DSL-like language again, it should be a
**signal expression language**, not a strategy DSL.

The intended scope is:

- represent one signal compactly
- support canonicalization and mutation
- support signal-level search and discovery

The intended non-scope is:

- full strategy authoring
- evaluation spec authoring
- workflow orchestration

In other words:

- signal expression language = good fit
- strategy DSL = low-value by default

The old `alpha_os_recovery.dsl` package was a retired legacy implementation.
It is not the source-of-truth model for the future system.

## First Principles

### The Scarce Resource

The scarce resource is not raw compute.

The scarce resources are:

- independent views of the world
- persistent structure rather than noise
- capacity to reduce many weak views into a smaller set of useful beliefs
- human ability to understand failure and concentration

So the system should optimize for:

- discovery breadth
- redundancy control
- failure visibility
- compressibility into action

It should not optimize for:

- preserving individual signal identity forever
- fully materializing every possible candidate
- treating all candidates as equally deserving of full validation

### Core Separation

The clean separation is:

```text
observation -> representation -> signal space -> selection -> compression -> decision
```

These are different problems and should remain different layers.

## System Boundary

`signal-noise` and `alpha-os` should remain independent.

`signal-noise` should be:

- an observation platform
- responsible for collection, normalization, storage, and delivery

`alpha-os` should be:

- an observation consumer
- responsible for representation, discovery, selection, compression, and decision

Neither system should know the internal implementation details of the other.

The shared surface should be an observation contract only.

## Greenfield Domain Objects

### Subject

A `subject` is an allocatable object.

Examples:

- a spot asset
- an equity
- an ETF
- an index
- a basket
- a sleeve
- a spread

`subject` is a portfolio concept, not a data-collection concept.

### Observable

An `observable` is a semantic data requirement.

Examples:

- `daily_close`
- `daily_return`
- `daily_volume`
- `realized_vol_20d`
- `spread_bps`

An observable is not:

- a collector name
- a backend table
- a series alias

### Observation Contract

The observation contract should answer:

- is this observable available for this subject or asset
- at what resolution
- with what freshness and provenance
- how to fetch the data

The consumer should not know:

- collector names
- storage schema
- backend signal aliases

### Representation

Representation is the layer that converts observations into reusable state.

Examples:

- feature matrices
- rolling statistics
- normalized cross-sectional views
- latent representations

This layer should be shared across large numbers of signals.

It should be the primary unit of computation.

### Signal Family

A signal family is a parameterized way of viewing the world.

Examples:

- momentum
- reversal
- range position
- dispersion
- flow-pressure
- regime-conditioned momentum

The greenfield unit is not an individual `signal_id`.

The greenfield unit is:

- family
- parameter space
- applicability constraints
- required observables

### Signal Discovery Spec

A signal discovery spec is the set of admissible
family/parameter/observable combinations used to generate candidate
signals.

It should encode:

- what can be generated
- what is allowed to run
- what observables are required
- what subject kinds are eligible

The signal discovery should be explicit.

It is a research object, not automatically the same thing as a trading
strategy.

New greenfield code should avoid the bare term `discovery` when the
distinction matters. The preferred terms are:

- `signal discovery`
- `strategy discovery`
- `adaptive discovery policy`

### Strategy Spec

A strategy spec is a concrete executable trading configuration.

It may include:

- universe
- signal family mix
- ranking or aggregation rule
- rebalance policy
- sizing method
- risk and turnover controls

A strategy spec may be authored directly, or it may be materialized from a
signal discovery process.

### Adaptive Discovery Policy

Some strategies may contain internal discovery.

Examples:

- generate fresh signals each month
- re-rank family members each quarter
- adaptively choose which predictive sleeve is active

In these cases, discovery is part of the strategy, but the adaptive discovery policy
should still be modeled separately from signal discovery itself.

## Terminology Boundary

The clean boundary is:

- `signal discovery`
  - the admissible region for generating and screening signals
- `strategy spec`
  - the executable trading configuration we actually evaluate and may deploy
- `strategy discovery`
  - the admissible region for comparing candidate strategy specs
- `adaptive discovery policy`
  - a strategy-internal discovery mechanism that may refresh signals over time

These are related concepts, but they are not interchangeable and should not
share a single overloaded source-of-truth name.

### Application Result

The true combinatorial object is not the subject or the signal alone.

It is the application result:

```text
subject × signal-family-parameter × time
```

This is where explosion occurs.

So this layer should be:

- lazily materialized
- screened early
- aggressively reduced

### Screening Result

A screening result is a cheap, first-pass judgment.

It should answer:

- should this candidate continue
- is it novel enough
- is it degenerate
- is it obviously unstable

Screening should be staged and intentionally cheap.

Cheap pre-screen exists to reject obvious losers early.

It does not exist to identify final winners.

So the greenfield rule is:

- cheap pre-screen may remove candidates that are clearly invalid, degenerate,
  unsupported, or obviously weak
- cheap pre-screen should tolerate false positives
- cheap pre-screen should try to avoid false negatives

Cheap pre-screen is allowed to use:

- observable availability
- applicability constraints
- sample coverage
- degenerate or near-constant signal checks
- very cheap predictive proxies
- obvious intra-family duplication

Cheap pre-screen should not try to decide:

- true stability under changing conditions
- redundancy after full survivor comparison
- portfolio contribution
- final model confidence
- final belief compression

Those belong to later planes.

### Compressed Belief

A compressed belief is what survives large-scale discovery.

It may be:

- an ensemble
- a cluster representative
- a latent factor
- a posterior belief over model families

Portfolio decision should consume compressed beliefs rather than raw large-scale
signal populations.

## Planes

### Observation Plane

Responsibilities:

- observation availability
- retrieval
- freshness and provenance

Primary system:

- `signal-noise`

### Representation Plane

Responsibilities:

- reusable feature construction
- reusable state computation
- shared feature caches

Primary unit:

- subject-by-time representation

### Signal Discovery Plane

Responsibilities:

- family definitions
- parameter grids or generators
- applicability rules
- signal-discovery compilation

Primary unit:

- family plus parameter space

### Selection Plane

Responsibilities:

- cheap filtering
- novelty and redundancy checks
- stability heuristics
- signal pruning

Primary unit:

- screening result

The first stage of this plane is cheap pre-screen.

Cheap pre-screen should be:

- computationally inexpensive
- representation-local
- mostly family-local
- safe to run before full persistence

Later stages of this plane may use:

- full evaluation outputs
- stability evidence
- redundancy comparison across survivors
- cross-subject and cross-family structure

### Compression Plane

Responsibilities:

- clustering
- factorization
- belief aggregation
- conversion from many weak views into fewer usable beliefs

Primary unit:

- compressed belief

### Decision Plane

Responsibilities:

- combine compressed beliefs with
  - estimation uncertainty
  - model uncertainty
  - structural uncertainty
  - dependence
  - cost
  - portfolio state
- choose desired portfolio state

Primary unit:

- portfolio intent

### Evaluation Plane

Responsibilities:

- map failure surfaces
- measure fragility
- test robustness of selection and compression
- test portfolio usefulness of compressed beliefs

Validation should not be reduced to:

- winner picking
- single scalar ranking

## Scaling Implications

### Ten Thousand Subjects, Ten Million Signals

At this scale:

- full materialization is the wrong abstraction
- one-row-per-candidate persistence is the wrong abstraction
- equal treatment of all candidates is the wrong abstraction

The system must assume:

- most candidates are redundant
- most candidates will never deserve full validation
- the final decision layer should only see a tiny compressed subset

### Required Computational Shape

The scaling shape should be:

- fetch observations once
- build shared representations once
- evaluate many family members in batch
- screen aggressively
- compress aggressively
- validate only survivors more deeply

Not:

- fetch per signal
- persist every candidate as first-class runtime truth
- full backtest every candidate

## What Must Be Avoided

The following are anti-patterns in the greenfield target:

- subject-specific signal definitions as the primary abstraction
- backend signal aliases leaking into the consumer contract
- treating every candidate as a durable entity
- making portfolio decision depend on raw large-scale candidate populations
- using full validation as the first filter

## Current Repo Gap Map

The current repository should be judged against this target.

### Already Aligned

- subject-first allocation concepts
- subject sets
- first-class observables
- observation-provider separation
- shared feature planes
- batch family execution

### Transitional

- executable signals as durable runtime objects
- full persistence of many evaluation rows
- full backfill as the default path
- validation that still behaves like broad replay rather than staged selection

### Missing

- first-class signal discoveries
- first-class screening results
- first-class compression outputs
- portfolio decision that consumes compressed beliefs instead of raw large
  candidate sets
- explicit redundancy reduction as a primary design layer

## Design Rule

When implementation choices are evaluated, the key question should be:

- does this move the system toward discovery, selection, and compression
  over signal space

not:

- does this make the current per-signal runtime slightly more convenient
