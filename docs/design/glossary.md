# Glossary

This file is the source of truth for core terminology in `alpha-os`.

If another design note uses a term differently, prefer this file.

## Core Terms

### portfolio

A set of holdings and their weights.

### universe

A defined set of instruments, assets, securities, or markets considered for investment, trading, or analysis.

### backtest

Testing a strategy or rule on historical data.

Backtests can be in-sample, out-of-sample, or walk-forward depending on the split.

### execution

The process of converting trading decisions into orders and fills.

### subject

An alpha-os internal term for one thing that can carry portfolio weight, such as
an asset, instrument, basket, sleeve, pair, or spread.

Examples:

- `BTC`
- `BTC_spot`
- `ETH_BTC_pair`
- `REIT_basket`
- `defensive_sleeve`

A subject may be backed by one or more instruments, but it is not identical to
an instrument.

### observable

A semantic data requirement that can be requested from `signal-noise`.

Examples: `daily_close`, `daily_return`, `realized_vol_20d`.

### feature

A reusable derived data series built from observables.

Features are representation-layer inputs to signals.

### representation

Shared computed state derived from observables and features.

Examples: rolling statistics, normalized panels, latent state.

### signal

A single predictive logic that consumes features or observables and produces predictions.

A signal is a predictive unit, not a full trading strategy.

### signal expression

A compact compositional representation of a signal.

This is a signal-level representation, not a strategy-level one.

### signal expression language

A language for authoring signal expressions.

If introduced, it should target signals only, not full strategies.

### signal family

A parameterized class of related signals.

Examples: momentum, reversal, range position.

### signal discovery

A research object that defines the admissible space for generating and screening signals.

It is not automatically the same thing as a strategy.

### thesis

A human-written market claim that motivates research.

It should express a claim, mechanism, scope, and failure modes without forcing implementation too early.

### research spec

A structured research document that translates one or more theses into signal and strategy research.

It is the bridge from idea generation into discovery and validation.

### experiment plan

One concrete offline test plan derived from a research spec.

It defines what is fixed, what varies, and how success is judged.

### execution record

An append-only record of what experiment was run and what happened.

It should link back to one experiment plan rather than rewriting the thesis.

### prediction

The concrete output a signal produces for a given subject and timestamp.

This is what signal evaluation scores. It is part of `SignalPolicy`, not a full trading strategy.

### signal contribution

A signal-level input to belief synthesis after screening and prediction orientation.

It records prediction, confidence, and marginal signal contribution.

### belief

A compressed or aggregated predictive view formed from multiple signals or predictions.

Belief is downstream of signals and upstream of decisions.

### sizing method

The portfolio weighting method used after ranking or belief formation.

Examples: `signal_weighted`, `equal_weight`, `risk_budgeting`, `minimum_variance`.

### trading strategy

The top-level trading object that defines what should be traded and how it should be realized.

It contains scope, inputs, position rule, portfolio policy, rebalance friction, and execution policy.

### strategy scope

The domain over which a trading strategy is defined before policy logic is applied.

Current examples are `subject_set` and `target`.

### strategy requirements

Conditions an instrument, dataset, or market context must satisfy for a strategy to run.

Examples: daily close is available, funding rate is available, shorting is allowed, minimum history length is met.

### tradable universe

The set of instruments a strategy may hold or trade.

This can differ from instruments observed only as references or context.

### evaluation universe

The set of instruments included in a specific evaluation run.

This is an evaluation condition, not necessarily the full strategy capability.

### position rule

The strategy rule that turns inputs into subject-level eligibility, direction, or timing decisions.

Current examples are `constant_hold`, `dual_momentum_hold`, and `crypto_regime_momentum_hold`.

### portfolio policy

The strategy sub-policy that converts predictive inputs into desired portfolio state.

Includes selection, sizing, rebalance, and risk.

### selection policy

The portfolio sub-policy that decides which subjects are admitted into the desired portfolio.

Current examples are `selection` and `top_k`.

### sizing policy

The portfolio sub-policy that decides how admitted subjects are weighted.

Current examples are `sizing_method` and, in evaluation/runtime construction, `sizing_engine`.

### sizing engine

The calculation engine used to realize a sizing method in evaluation or runtime decision construction.

Examples: `rule_based`, `optimizer`, `history_based`. It is narrower than the strategy run engine.

### rebalance policy

The portfolio sub-policy that decides how often desired portfolio intent is refreshed.

Current example is `rebalance`.

### risk policy

The portfolio sub-policy that constrains desired portfolio exposure.

Current examples are `long_only` and `gross_exposure_cap`.

### rebalance friction policy

The strategy sub-policy that defines how current state should move toward desired state under rebalance frictions.

Includes turnover friction, no-trade band, and execution-cost aversion.

### execution policy

The strategy sub-policy that defines how desired portfolio state should be realized.

Includes urgency, order style, slicing, and venue-facing constraints.

### strategy

A complete executable trading specification.

It may include universe, signals, ranking, allocation, rebalance, and risk controls.

### strategy spec

A concrete structured definition of a strategy.

Defines trading behavior. Current mainline `TradingStrategySpec` is the first-class strategy model.

### strict OOS evaluation

The primary evaluation mode for comparing a strategy under train/test separation.

This is the default evaluation-side request shape.

### evaluation spec

The rules for how a strategy is evaluated.

Defines the measurement recipe: strict OOS, fold layout, costs, metrics.

### evaluation task

One executable evaluation defined by `strategy spec + evaluation spec`.

Binds one strategy to one evaluation spec as a concrete run setup, including
required train or strategy checkpoint artifacts.

### data input

The logical data input used by evaluation or research.

A bounded dataset for offline evaluation or a stream for online evaluation.

### data source

The runtime connection source used to read data.

`base_url` belongs here, not on evaluation tasks.

### strategy checkpoint

A strategy checkpoint used to execute a strategy in evaluation.

It may come from a train period or another checkpoint source.

### signal train

A shared upstream training unit used to produce signal-related state.

Only strategies that require training need this concept.

### train artifact

A frozen output from train data later applied in test without re-selection.

Examples: selected signals, fitted compression settings.

### checkpoint-based evaluation

Running evaluation with a precomputed strategy checkpoint instead of retraining.

This is a comparison shape, not the default evaluation shape.

### evaluation task result

The recorded factual result of one evaluation task.

Includes metric group results, failure finding groups, and artifact references.

### evaluation report

A persisted record container for one or more evaluation task results. It is not a comparison object.

Includes task results from one evaluation run.

### evaluation metric group

An evaluation category / metric group.

Examples: `decision_quality`, `cost_drag`, `portfolio_target_return_alignment`.

### evaluation metric

One concrete scalar measurement inside an evaluation metric group.

Example: `portfolio_target_return_corr`.

### evaluation metric group result

One result block for one evaluation metric group.

`metric_group_name + source + metrics`.

### evaluation metric group name

The identifier for an evaluation metric group when a contract references metric fields.

`metric_group_name="decision_quality"`

### benchmark

A reference index, portfolio, or strategy used for comparison.

Examples: S&P 500, TOPIX, MSCI World, 60/40 portfolio, equal-weight portfolio.

### net return

Return after modeled costs and frictions.

Use gross return when costs and frictions are excluded.

### drawdown

Decline from a prior peak in portfolio value or cumulative return.

Maximum drawdown is the worst such decline over a period.

### Sharpe ratio

Return per unit of return volatility.

The annualization and risk-free-rate convention must be stated by the metric producer.

### turnover

The amount of portfolio weight or exposure changed over a period.

Turnover convention must be stated when comparing costs or strategies.

### alpha

Excess return over a benchmark.

`alpha` is an outcome, not a predictive unit.
Short implementation notes:

- `EvaluationTaskResult` is the class name for an evaluation task result. Use
  `EvaluationReport.task_results` in runtime readers and serialization because
  `task_results` is the persisted report field.
- Runtime readers and serialization should use `metric_group_results`.
- One evaluation task has exactly one strategy; one strategy may appear in many
  cases. Runtime case roles are limited to `baseline` and `standard`.
- Candidate/diagnostic-style labels belong in research notes, not manifests or
  reports.
- `Benchmark` is intentionally not an evaluation-settings class name. Reserve it
  for market indexes, benchmark portfolios, and benchmark-relative return/risk
  measurement.

## Term Boundaries

### Signal vs Strategy

A **signal** is a predictive component.

A **strategy** is an executable trading configuration that may use one or more
signals plus additional logic such as:

- universe selection
- ranking or aggregation
- sizing
- rebalance policy
- risk controls

So:

- signal = predictive unit
- strategy = trading system

### Trading Strategy vs Run Context

- **trading strategy** = what should be traded and how it should be realized
- **run context** = the engine context and inputs for running that trading
  strategy

So:

- `trading strategy` belongs to strategy semantics
- required run inputs belong to engine context

### Strategy Scope vs Position Rule

- **strategy scope** = where the strategy is defined
- **position rule** = how strategy inputs become subject-level position intent

So:

- `subject_set` and `target` belong to strategy scope
- `position_rule_id`, `signal_discovery_id`, and `family_mix` describe the
  position rule or the artifacts it depends on
- `execution_kind` is a transitional implementation field, not a glossary term

### Strategy Requirements vs Universe

- **strategy requirements** = conditions that must hold for a strategy to run
- **tradable universe** = what the strategy may hold or trade
- **evaluation universe** = what an evaluation run measures

Strategy inputs are not automatically a universe. Observed data belongs first to
the signal, feature, or observable layer unless it defines what can be traded or
what is included in an evaluation.

### Portfolio Policy vs Rebalance Friction Policy vs Execution Policy

- **portfolio policy** = what portfolio state should be desired
- **rebalance friction policy** = how aggressively current state should move toward that desired state under rebalance frictions
- **execution policy** = how that transition should be realized in the market

So:

- sizing and rebalance belong to portfolio policy
- turnover friction and no-trade band belong to rebalance friction policy
- slippage model and order style belong to execution policy
- cost and friction settings should be expressed explicitly in rebalance friction and execution policy

Inside portfolio policy:

- selection policy decides admission and concentration
- sizing policy decides weighting
- rebalance policy decides refresh cadence
- risk policy decides exposure limits

### Strategy vs Strategy Run

- **strategy** = what the trading logic means
- **strategy run** = how an engine runs that strategy in a specific context

Different engines may optimize mechanics, but they should not change strategy
semantics.

### Strategy Run vs Evaluation

- **strategy run** = the broader concept
- **evaluation** = one engine context that runs a strategy for measurement

So `evaluation` is acceptable when scoped as:

- `evaluation spec`
- `evaluation task`
- `evaluation report`

Bare `evaluation` should not become the universal name for all strategy
execution.

### Signal Expression Language vs Strategy Spec

A **signal expression language** is a representation for a signal.

It may be useful because signals are:

- small
- compositional
- easy to canonicalize
- suitable for mutation and search

A **strategy spec** is broader and should usually remain structured data rather
than a text DSL.

So:

- signal expression language = notation for signals
- strategy spec = structured executable trading definition

### Signal Discovery vs Strategy Discovery

**signal discovery** defines what signals may be generated and screened.

**strategy discovery** defines what strategy specs may be compared.

These are related, but they are not interchangeable.

### Thesis vs Research Spec vs Experiment Plan

- **thesis** = what might be true in the market
- **research spec** = how that idea should be expressed as research inputs
- **experiment plan** = what concrete offline test will be run now
- **execution record** = what was actually run and observed

These objects form a lifecycle, not a ranking tree.

Many theses may coexist at the same time.

### Signal vs Prediction vs Belief

- **signal** = the predictive logic
- **prediction** = one output produced by that signal
- **belief** = an aggregated view built from many predictions or signals

These are all below the `TradingStrategy` level.

- `SignalPolicy` may produce predictions
- beliefs may aggregate those predictions
- portfolio, rebalance, execution, and adaptation still remain separate
  strategy layers

### Observable vs Feature

- **observable** = semantic raw requirement from `signal-noise`
- **feature** = reusable derived series built on top of observables

## Legacy Mapping

The following legacy terms should not be used as new source-of-truth names:

| Legacy Term | Use Instead |
|------------|-------------|
| `alpha` as predictive unit | `signal` |
| `signal candidate` | `signal` |
| `dsl` as a bare name | `signal expression language` when the distinction matters |
| `search` when meaning generated-and-screened signal space | `signal discovery` |
| `experiment` as universal comparison object | usually `evaluation task` or `evaluation report`, depending on context |

## Naming Rule

When introducing a new object:

1. Use **signal** for predictive units.
2. Use **strategy** for executable trading specs.
3. Use **discovery** for admissible generation-and-screening spaces.
4. Reserve **alpha** for outcome metrics only.
