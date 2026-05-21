# alpha-os Constitution

## North Star

alpha-os is a research-to-evaluation OS for trading strategies.

Its job is to move an investment idea through a reproducible lifecycle:

```text
idea -> exploratory research -> candidate -> strict OOS evaluation
     -> promotion or rejection -> baseline
```

alpha-os should not be only a backtester, only a signal generator, only a
runtime trader, or only a document archive. Those are components. The system is
the lifecycle that turns uncertain market ideas into evaluated decisions.

## Lifecycle Stages

### Idea

An idea is a market claim. It may be incomplete and may use informal language.
It should answer what might work, where it might work, and why it is worth a
test.

### Exploratory Research

Exploratory research is allowed to be rough, but the roughness must be explicit.

Allowed roughness includes:

- provisional universes
- simple cost assumptions
- external benchmark proxies
- incomplete instrument coverage
- coarse parameter grids
- lightweight scripts or manifests

Exploration must optimize for speed, comparison, and discardability. It should
not pretend to be final evidence.

### Candidate

A candidate is an idea that survived exploration.

Before promotion to strict OOS, it must have:

- a named strategy hypothesis
- a fixed universe or universe construction rule
- fixed signal and allocation recipes
- explicit cost and data assumptions
- a baseline to beat
- a decision rule that can reject it

### Backtest OOS Evaluation

Backtest OOS is the adoption gate, not the first place every idea must run.

Backtest OOS requires:

- fixed folds and date ranges
- reproducible inputs
- explicit portfolio construction
- cost, funding, roll, borrow, and multiplier treatment when relevant
- run result and decision trace persistence
- enough diagnostics to explain failure, not only aggregate metrics

### Promotion Or Rejection

Promotion requires evidence against the current baseline, not just positive
absolute performance.

A promoted strategy should improve the baseline on a defensible combination of:

- compounded OOS return
- drawdown and worst-fold behavior
- fold stability
- cost robustness
- turnover realism
- diversification value

Rejection is a useful result. Rejected ideas should not keep accumulating code
unless they expose a reusable system gap.

### Baseline

A baseline is the current strategy to beat. It is not sacred, but it is the
default alternative.

Changing the baseline requires evidence. Adding complexity without beating the
baseline is not progress.

## Rigor Levels

alpha-os should support multiple rigor levels as first-class states, not as
side exits.

```text
exploratory  -> fast, rough, disposable
candidate    -> fixed assumptions, comparable
backtest_oos   -> reproducible adoption evidence
operational  -> live or paper-runtime eligible
```

The mistake to avoid is forcing exploratory work through strict OOS too early,
or allowing strict OOS claims to rest on exploratory shortcuts.

## Strategy Research Principles

Do not search for a single magic strategy.

Systematic macro and managed futures research should compare families:

- trend across multiple horizons
- carry
- value
- defensive or volatility-aware overlays
- cross-sectional momentum
- blended sleeves
- portfolio construction variants

The system should make it easy to compare these families under shared metrics
and baselines.

Parameter search is allowed only when robustness is checked. A strategy that
works only at one fragile parameter setting should be treated as unproven.

## Evaluation Principles

The system must distinguish these questions:

- Is the market hypothesis weak?
- Is the data incomplete or distorted?
- Is the portfolio construction wrong?
- Is the evaluation too narrow?
- Is the implementation buggy?

Aggregate Sharpe or return is not enough. Evaluation should preserve enough
trace and attribution to separate signal direction, allocation, costs,
turnover, long/short behavior, and asset-class contribution.

## System Boundaries

### alpha-os

alpha-os owns:

- strategy lifecycle
- signal and portfolio recipes
- evaluation specs
- strict OOS reports
- decision traces
- promotion and rejection evidence

### signal-noise

signal-noise is a separate data service.

alpha-os should not push alpha-os-specific product assumptions into
signal-noise. alpha-os owns adapters, contracts, and validation of what it
needs from external data. signal-noise owns its own data product.

### Research Artifacts

Research artifacts exist to speed decisions, not create process drag.

The default should be lightweight. More structure is justified only when it
prevents future ambiguity or makes a comparison reproducible.

## Documentation Roles

Design documents under `docs/design/` hold durable architectural decisions.

Generated reports and committed manifests are factual records. Hand-written
documents should not duplicate generated evidence unless they add interpretation
or a decision.

## Non-Negotiables

- Do not confuse exploration evidence with strict OOS evidence.
- Do not optimize implementation convenience over market hypothesis clarity.
- Do not add strategy complexity unless it targets an observed failure mode.
- Do not let documentation become a substitute for running evaluations.
- Do not treat design patterns as goals. They are tools for lifecycle clarity.
- Do not couple alpha-os to signal-noise internals.
- Do not promote a strategy without a baseline comparison and rejection rule.

## Practical Decision Rule

When a proposed change appears, classify it first:

```text
Does it improve exploration speed?
Does it improve candidate comparability?
Does it improve strict OOS credibility?
Does it improve operational safety?
```

If it improves none of these, it is probably local cleanup or process drag.

If it improves one of them while harming another, make the tradeoff explicit
before implementing.
