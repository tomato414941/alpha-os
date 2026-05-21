# Strategy Run Model

This file is the source of truth for how a `strategy spec` runs across different
engines.

It separates:

- trading-strategy semantics
- engine mechanics
- current mainline workflow
- target long-horizon workflow

## Core Rule

A trading strategy should be portable across engines.

That means:

- the same `trading strategy` should mean the same thing everywhere
- different engines may optimize how it is run
- engines should not silently change what the strategy does

So:

- **semantics must stay consistent**
- **mechanics may differ by engine**

## Trading Strategy vs Engine

- **trading strategy**
  - the portable trading definition
  - composed of strategy scope, inputs, position rule, portfolio policy, rebalance friction policy, and execution policy
- **engine**
  - the mechanism that runs the strategy in a specific context

Examples of engines:

- strict OOS evaluation engine
- checkpoint evaluation engine
- paper engine
- live engine

`evaluation` is slightly too broad as a bare noun, but the current mainline
names are acceptable when they stay scoped:

- `evaluation spec`
- `evaluation case`
- `evaluation run result`

These should be read as **trading-strategy evaluation** concepts, not as generic
execution concepts.

## Trading Strategy Hierarchy

The long-horizon target hierarchy is:

```text
TradingStrategy
├─ StrategyScope
├─ SignalPolicy
│  ├─ SignalDefinitionPolicy
│  └─ SignalUpdatePolicy
├─ PortfolioPolicy
│  ├─ SelectionPolicy
│  ├─ SizingPolicy
│  ├─ RebalancePolicy
│  └─ RiskPolicy
├─ RebalanceFrictionPolicy
└─ ExecutionPolicy
```

This is distinct from run context:

```text
StrategyRunSpec
├─ TradingStrategy
└─ Explicit run inputs
```

So:

- `TradingStrategy` defines what should be traded and how it should be realized
- explicit run inputs define the engine-side context and artifacts required to
  run it

`prediction` is not a peer of `TradingStrategy`. It is one output produced
inside `SignalPolicy`.

So the clean relation is:

- `prediction` = one signal-layer output
- `signal evaluation` = whether those outputs contain usable information
- `strategy evaluation` = whether the full trading strategy produces good
  portfolio outcomes

Good predictions do not guarantee a good strategy. Good strategy outcomes also
do not prove that prediction quality was the dominant source of edge.

## Current State Handling

The mainline trading-strategy contract no longer stores `execution_kind`.
Evaluation state sourcing is now represented by explicit inputs:

- direct strategy evaluation uses the strategy scope and current inputs
- checkpoint evaluation uses `strategy_checkpoint_id` or fold checkpoints

### Checkpoint-Based Evaluation

Use when evaluation should reuse a precomputed fixed strategy checkpoint rather
than retraining during evaluation.

Properties:

- a fixed strategy checkpoint is required
- replay can compare downstream behavior without re-running discovery
- this is the preferred benchmark shape when the upstream state should stay fixed

## Current Evaluation Job Shapes

The current mainline engine contract no longer stores `run_mode` on evaluation
job specs. Evaluation behavior should be represented through explicit inputs.

- `backtest_oos`
  - run the strategy under a strict train/test-separated evaluation spec
- `paper`
  - reserved for future paper-trading execution
- `live`
  - reserved for future live execution

So:

- `execution_kind` has been removed from trading strategy specs
- `run_mode` has been removed from evaluation job specs
- future strategy requirements should be expressed directly instead of hidden
  behind a single mode-like term
- future evaluation job shapes should make required inputs explicit instead of
  hiding them behind a generic mode

## Current Mainline Mapping

The current codebase is still transitional. The practical mapping is:

| Current object | Closest long-horizon concept | Notes |
|---------------|------------------------------|-------|
| `TradingStrategySpec` | `TradingStrategy` | First-class structured strategy definition used by current mainline. |
| `execution_kind` | removed implementation field | Strategy specs and evaluation planning no longer use it. |
| `run_mode` | removed implementation field | Evaluation job specs now express required inputs directly. |
| evaluation case tuple | partial `StrategyRunSpec` | It binds a result key to a strategy id for one evaluation run. |
| `EvaluationSpec` | evaluation measurement recipe | It is not a generic run-policy object. |

### Current Evaluation Job Shapes

| Evaluation job shape | Purpose | Required inputs | Retraining during evaluation |
|----------------------|---------|-----------------|------------------------------|
| `backtest_oos` | Evaluate the strategy under train/test separation. | `evaluation case`, `evaluation spec`, and any strategy-side train artifacts needed by the strategy. | Allowed when the strategy requires train-period state. |
| checkpoint-based evaluation | Compare downstream behavior while holding upstream state fixed. | `evaluation case`, `evaluation spec`, `strategy_checkpoint_id`. | Never. |

This means:

- `backtest_oos` is the primary evaluation mode
- checkpoint-based evaluation is a comparison shape, not the default evaluation mode
- the difference is owned by the engine contract, not by strategy semantics

The evaluation object remains the trading strategy under a specific engine
context. Prediction-level metrics are only one part of the resulting run result.

### Current Run-Input Contracts

The current mainline treats these inputs as separate contract objects.

| Evaluation job shape | Contract object | Required fields |
|----------------------|-----------------|-----------------|
| `backtest_oos` | `BacktestOosRunInputs` | `evaluation_spec_id`, `execution_range`, `evaluation_date_ranges`, `metric_group_names` |
| checkpoint-based evaluation | checkpoint evaluation inputs | `evaluation_spec_id`, `strategy_checkpoint_id`, `execution_range`, `evaluation_date_ranges`, `metric_group_names` |

## Current Mainline Workflow

The current mainline has three distinct workflows.

### 1. Signal Discovery Run

Purpose:

- generate and screen signals
- produce train-period outputs
- create `strategy checkpoint` when needed

Inputs:

- `signal discovery`
- subject set
- target
- train-period data

Outputs:

- screening result
- compressed belief
- strategy checkpoint

Use this when:

- the strategy needs upstream discovery or fitting
- a new trained strategy checkpoint must be produced

### 2. Evaluation Run

Purpose:

- compare a strategy under an evaluation spec
- run strict OOS folds

Inputs:

- `evaluation case`
- evaluation spec
- explicit evaluation job inputs

Outputs:

- evaluation run result
- evaluation metric group results
- fold-level runtime artifacts

Use this when:

- the question is whether a strategy performs well under strict OOS
- retraining during evaluation is part of the allowed evaluation procedure

### 3. Fixed-State Replay

Purpose:

- compare downstream strategy behavior while holding upstream state fixed

Inputs:

- `evaluation case`
- fixed strategy checkpoint
- evaluation spec

Outputs:

- evaluation run result
- replay-only comparison metrics

Use this when:

- the upstream signal state is already known
- the comparison question is about downstream strategy behavior
- re-running discovery would add cost without adding insight
- retraining during evaluation would blur the comparison question

## Current Mainline Boundary

The current mainline should be understood like this:

- discovery finds or fits signal-related state
- evaluation compares strategies
- checkpoint-based evaluation compares strategies without re-running discovery

So:

- discovery does **not** directly decide portfolio outcomes
- evaluation does **not** invent new signals during replay
- checkpoint-based evaluation does **not** retrain

## Target Workflow

The target long-horizon workflow is slightly broader than the current mainline.

### 1. Signal Discovery

- find useful signals
- evaluate and screen them

### 2. Strategy Construction

- build executable strategies from signals, selection, allocation, rebalance,
  and risk rules

### 3. Backtest OOS Evaluation

- compare strategies under shared evaluation specs

### 4. Promotion

- promote strong strategies into standard or production-ready candidates

## Why Current And Target Are Different

The current mainline is an implementation-oriented operating model.

The target workflow is the cleaner conceptual model.

They overlap, but they are not identical:

- the current mainline explains how the codebase runs today
- the target workflow explains what the architecture should converge toward

## Design Rule

When changing execution behavior, ask:

1. does this preserve strategy semantics across engines
2. does this change only mechanics, not meaning
3. does this make discovery, evaluation, and replay easier to reason about as
   separate workflows

If the answer is no, the change is probably mixing engine mechanics back into
strategy semantics.

## Current Code Boundary

The current codebase should converge on this split:

- `TradingStrategySpec`
  - strategy semantics
- `StrategyExecutionRequest`
  - current implementation name for one engine-specific run spec
- evaluation case tuple
  - transient result key and strategy id selected for one run
- `EvaluationRunResult`
  - the recorded result of strategy evaluation

In other words:

- not every strategy run is an evaluation
- but every current mainline evaluation is a form of strategy run
