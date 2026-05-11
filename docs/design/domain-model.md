# Domain Model

Terminology definitions live in [`glossary.md`](./glossary.md).
This file focuses on relationships between domain objects.

## Core Terms

| Term | Definition | Examples |
|------|-----------|---------|
| **feature** | Input data series from signal-noise. Raw or computed market observables that signals consume. | `fear_greed`, `btc_ohlcv`, `vix_close`, `funding_rate_btc` |
| **signal** | A single predictive logic that consumes features and produces predictions. A claim about market inefficiency; it may or may not have real predictive power. | `(sub fear_greed dxy)`, XGBoost model, RSI mean-reversion rule |
| **prediction** | The concrete output value a signal produces for a given date and asset. This is what the pipeline evaluates. | `+0.3`, `-0.15` |
| **alpha** | Excess return over benchmark. It is an outcome, not a method. | Sharpe 0.5 after subtracting benchmark |

The primary data flow is:

`feature -> signal -> prediction`

The evaluation pipeline scores predictions, not signal internals. But trading
strategy evaluation is broader than prediction evaluation: predictions are one
input into portfolio, rebalance, execution, and adaptation decisions.

## Research Terms

| Term | Definition | Examples |
|------|-----------|---------|
| **research axis** | A dimension along which candidate strategies can differ. A strategy is not atomic; it is composed by assigning values on multiple axes. | universe, signal family mix, model method, rebalance policy, sizing method |
| **signal discovery** | Offline or research-time signal candidate search. It defines which signal candidates may be generated and evaluated, but it is not automatically the same thing as a strategy. | momentum/reversal families with admissible lookbacks and observables |
| **strategy discovery** | A research object that defines which strategy candidates may be compared. It ranges over executable strategy specs rather than over signals alone. | compare `crypto + equal_weight + weekly` against `multi_asset + HRP + annual` |
| **adaptive discovery policy** | A strategy-internal mechanism that re-generates or re-selects signals or sleeves over time. This is part of an adaptive strategy, not the same thing as signal discovery. | monthly family re-selection, rolling sleeve activation |
| **trading strategy** | The top-level trading object. It combines scope, inputs, position rule, portfolio policy, rebalance friction, and execution policy into one portable trading definition. | `ETF rotation + relative strength + equal weight + simple execution` |
| **position rule** | The strategy rule that turns inputs into subject-level eligibility, direction, or timing decisions. | `constant_hold`, `dual_momentum_hold`, `crypto_regime_momentum_hold` |
| **execution kind** | Legacy/transitional implementation wording. Do not use as a domain term. | Use explicit strategy requirements and run state sourcing instead. |
| **signal contribution** | A signal-level input to belief synthesis after screening and prediction orientation. It records prediction, confidence, and marginal signal contribution. | `SignalContribution(signal_id="trend@AAPL", ...)` |
| **belief synthesis** | The process that combines signal contributions into target-level belief components. | cluster related signal families, compute belief confidence |
| **compressed belief** | A compact belief artifact produced from belief synthesis for downstream portfolio decisions. | `CompressedBeliefComponent(signal_contribution_count=3, ...)` |
| **portfolio policy** | The strategy sub-policy that turns predictive inputs into desired portfolio state. | selection, sizing, rebalance, risk |
| **rebalance friction policy** | The strategy sub-policy that defines how current portfolio state should move toward desired portfolio state under rebalance frictions. | turnover friction, no-trade band, execution-cost aversion |
| **execution policy** | The strategy sub-policy that defines how desired state should be realized. | urgency, order style, slicing, venue-facing limits |
| **strategy spec** | A concrete structured strategy definition used by the runtime. It defines trading behavior. | current mainline `TradingStrategySpec` |
| **strategy** | A complete executable trading specification. In clean long-horizon terminology this is a trading strategy. | `multi_asset_full_universe + weekly rebalance + HRP`; `ETF rotation + relative strength + equal weight` |
| **strategy run** | Running a strategy through a specific engine context. | strict OOS evaluation, frozen replay, paper, live |
| **run policy** | The engine-side policy that selects the run context for a strategy. | `backtest_oos`, `fixed_state_replay`, `paper`, `live` |
| **strategy run spec** | One trading strategy paired with one run policy. | strategy under strict OOS; strategy under paper mode |
| **evaluation spec** | The rules for how a strategy is evaluated. It defines the measurement recipe. | fold layout, costs, metric windows |
| **evaluation task** | One executable evaluation defined by `strategy spec + evaluation spec`. It binds one strategy to one evaluation spec as a concrete run setup, including `signal_train`, run mode, and fixed state. | one strategy under one strict OOS evaluation spec |
| **data input** | The logical data input used by an evaluation or research run. It may be a bounded dataset or an online stream. | fixed global macro dataset; broker paper feed |
| **data source** | The runtime connection source used to read data inputs. Connection details are runtime configuration, not evaluation task metadata. | signal-noise service URL; local parquet root |
| **evaluation task result** | The recorded factual outcome of one evaluation task. | OOS Sharpe, belief corr, turnover, drawdown |
| **evaluation report** | A persisted record container for one or more evaluation task results. It is not a comparison object. | task results from one evaluation run |
| **evaluation metric group** | An evaluation category / metric group requested by an evaluation spec and recorded in an evaluation report. | `decision_quality`, `cost_drag`, `portfolio_target_return_alignment` |
| **evaluation metric** | One concrete scalar measurement inside an evaluation metric group. | `mean_decision_net_return`, `portfolio_target_return_corr` |
| **evaluation metric group result** | One result block for one evaluation metric group. It is `metric_group_name + source + metrics`. | `EvaluationMetricGroupResult(metric_group_name="decision_quality", metrics={...})` |
| **evaluation metric group name** | The identifier for an evaluation metric group when a contract references expected metric fields. | `CrossInstrumentMetricContract.metric_group_name="decision_quality"` |
| **evaluation profile** | Legacy term. Do not use it for new code or docs. | Use `evaluation metric group result`. |
| **train artifact** | A frozen output produced from the train period and later applied to the test period without re-selection. It is the evaluation-time analogue of a fitted model artifact. | survivor signal set, fitted belief synthesis settings, frozen allocation parameters |

## Evaluation Result Terminology

`EvaluationMetricGroupResult` is one result block for one evaluation metric group.
The old profile-oriented implementation name has been removed.
Read `evaluation metric group` as `evaluation category` or `metric group`; it is
the grouping key for related metrics such as decision quality, cost drag, or
concentration.
In report contracts, use `metric_group_name` for this identifier to avoid
confusing the metric group concept with the contract field name.

```text
EvaluationSpec = evaluation settings
EvaluationTask = strategy + evaluation settings
EvaluationTaskResult = recorded factual outcome of one evaluation task
EvaluationReport = persisted container of evaluation task results
EvaluationMetricGroupResult = one metric group result block inside a task result
```

Reserve `Benchmark` for trading comparison references such as market indexes,
benchmark portfolios, excess-return bases, and benchmark-relative risk
measurement. Do not use `Benchmark` for evaluation settings.

One `EvaluationTask` has exactly one strategy. A strategy may appear in many
evaluation tasks under different protocols or run setups.

Do not use `profile` to mean an evaluation configuration template. Use
`EvaluationSpec` for evaluation settings.

Evaluation reports should record and display measured facts. Relative
comparisons between findings are separate comparison views, not part of the core
evaluation result terminology.

The implementation name for an evaluation task result is `EvaluationTaskResult`.
Treat it as a task-level result record, not as a comparison row.
`EvaluationReport.task_results` remains the runtime and persisted field for
evaluation task results.

Use `EvaluationReport.task_results` in display, validation, diagnostics, and
analysis code.

Use `EvaluationTaskResult.metric_group_results` in runtime readers and persisted
documents.

## Signal And Strategy Boundary

Use deliberately explicit names around signal selection:

- `signal` = a prediction-producing component.
- `signal discovery` = offline or research-time signal candidate search.
- `position rule` = the strategy-internal rule that turns inputs into
  eligibility, direction, or timing decisions.
- `portfolio_target_return_alignment` = an evaluation metric group that measures
  whether portfolio target weights align with subsequent realized returns.

If a future strategy searches for or re-selects signals during paper or live
trading, model that as a strategy-internal runtime signal selection policy, not
as the same thing as offline signal discovery.

## Evaluation Case Semantics

An evaluation task is runtime setup, not research taxonomy. It does not carry a
case role or data-source URL.

`base_url` is runtime data-source connection configuration. It may flow through
execution requests and execution-plan entries, but it is not persisted on
`EvaluationTask`. The evaluation model should eventually refer to a `DataInput`
such as a fixed dataset or online stream, while the runtime maps that input to a
connection source.

Comparison anchors such as "baseline" are chosen by comparison views or
research notes. They are not fields on `EvaluationTask` or
`EvaluationTaskResult`.

Use candidate/diagnostic-style wording only in documents and research notes:

- candidate-like: a strategy or case being judged as a possible improvement
- diagnostic-like: an analysis probe used to explain why another case works or
  fails

These labels are useful for human discussion, but they are not manifest schema,
report schema, evaluation task roles, or strategy kinds.

Trading strategy is the first-class trading concept. It should usually include:

- strategy scope
- inputs
- position rule
- portfolio policy
- rebalance friction policy
- execution policy

In clean long-horizon design, that hierarchy is:

```text
TradingStrategy
├─ Scope
├─ Inputs
├─ PositionRule
├─ PortfolioPolicy
├─ RebalanceFrictionPolicy
└─ ExecutionPolicy
```

Run context should be modeled separately:

```text
StrategyRunSpec
├─ TradingStrategy
└─ RunPolicy
```

So:

- `trading strategy` = what should be traded and how it should be realized
- `run policy` = in what context it should be run

Strategy execution is broader than trading strategy. It additionally includes
the engine context in which the strategy is run, such as:

- evaluation spec
- paper or live runtime
- fixed-state replay versus retraining
- runtime-specific batching or caching

A trading strategy is what we want to define. A strategy run is how an engine
runs that trading strategy in a particular context.

## Current Mainline Mapping

The current repo is still converging, but the practical mapping is now direct:

| Current object | Closest long-horizon concept | Notes |
|---------------|------------------------------|-------|
| `TradingStrategySpec` | `TradingStrategy` | First-class structured strategy definition. |
| `execution_kind` | transitional implementation field | Remove once strategy requirements and run/evaluation state sourcing are represented directly. |
| `run mode` | `RunPolicy` | `backtest_oos` and `fixed_state_replay` are engine-side choices. |
| `EvaluationTask` | partial `StrategyRunSpec` | It binds a trading strategy to an evaluation context. |
| `EvaluationSpec` | evaluation-specific run-policy details | This is not the full run-policy universe; it is the evaluation branch of it. |

Bare `discovery` is too ambiguous for source-of-truth terminology.

New code and documentation should prefer one of:

- `signal discovery`
- `strategy discovery`
- `adaptive discovery policy`

## Strategy Hierarchy

alpha-os should treat `TradingStrategy` as the primary strategy object.

```
TradingStrategy
├─ StrategyScope
│  ├─ subject set
│  └─ target
├─ SignalPolicy
│  ├─ SignalDefinitionPolicy
│  └─ SignalUpdatePolicy
├─ PortfolioPolicy
│  ├─ SelectionPolicy
│  ├─ SizingPolicy
│  ├─ RebalancePolicy
│  └─ RiskPolicy
├─ RebalanceFrictionPolicy
│  ├─ turnover friction
│  ├─ no-trade band
│  └─ execution-cost aversion
└─ ExecutionPolicy
   ├─ slippage model
   ├─ execution objective
   ├─ order policy
   ├─ fill policy
   └─ venue policy
```

The platform may eventually hold multiple trading sleeves, but the current
mainline remains one predictive sleeve.

`subject_set` and `target` should be read as trading-strategy scope, not as
signal-policy fields. They define where the strategy is defined before any
signal-generation rule is applied.

Within `SignalPolicy`, the current clean split is:

- `SignalDefinitionPolicy`: which signal logic is used, such as `discovery`, `signal`, and `family_mix`
- `SignalUpdatePolicy`: how that signal logic is produced or reused, currently expressed by `execution_kind`

Within `PortfolioPolicy`, the current clean split is:

- `SelectionPolicy`: admission and concentration controls such as `selection` and `top_k`
- `SizingPolicy`: weighting choice such as `sizing_method`; evaluation/runtime
  construction may also choose a `sizing_engine`
- `RebalancePolicy`: intent refresh cadence such as `rebalance`
- `RiskPolicy`: exposure-side limits such as `long_only` and `gross_exposure_cap`

## Multi-Strategy Extensibility

The name `alpha-os` is goal-oriented rather than method-oriented. The platform
may later contain:

- predictive sleeves
- arbitrage sleeves
- market-making sleeves

Each sleeve owns its internal pipeline. The shared layer should provide:

- capital allocation across sleeves
- risk management
- execution infrastructure
- shared market data

Signals are evaluated by prediction quality. Sleeves are evaluated by
portfolio outcomes such as Sharpe, drawdown, and correlation.

So:

- `prediction` is part of `SignalPolicy`
- signal evaluation is part of strategy evaluation
- `TradingStrategy` remains the top-level trading object

## Signal Dimensions

A signal is defined by two orthogonal axes:

- **method**: how it is built
- **domain**: what market phenomenon it targets

### Method

| Method | What it does | Status |
|--------|-------------|--------|
| **DSL / GP** | Compose features via S-expression operators, evolved by genetic programming | Active |
| **Human-authored** | Fixed rules written by humans from domain knowledge | Active |
| **ML** | Learn patterns from features statistically | Planned |
| **LLM / NLP** | Extract predictions from unstructured text | Future |
| **Meta / ensemble** | Combine other signals' predictions | Future |
| **External ingest** | Import predictions from outside systems | Future |

### Domain

| Domain | What it targets | Examples |
|--------|----------------|---------|
| **Technical / macro** | Price patterns, macro indicators | RSI, momentum, carry |
| **Options** | Volatility surface, skew, term structure | IV smile interpretation |
| **Order flow** | Microstructure signals | VPIN, book imbalance |
| **On-chain** | Blockchain observables | Wallet flows, whale behavior |
| **Event** | Calendar and news events | FOMC, halving, earnings |
| **Lead-lag** | Cross-asset time delays | ETH lagging BTC |
| **External markets** | Prediction markets, analyst forecasts | Polymarket odds |

The same method can target different domains, and the same domain can be
approached with different methods.

## Terminology Constraint

`alpha` should be reserved for excess return as an outcome.

New code and documentation should prefer:

- `signal` for predictive units
- `prediction` for concrete outputs

Legacy names such as `alpha_id` remain only as compatibility residue and should
not become new source-of-truth concepts.
