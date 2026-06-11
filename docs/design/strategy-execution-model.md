# Strategy Run Model

Legacy design note. Do not treat this file as the current source of truth.

The current maintained contract is `TradingStrategy.decide(input) -> output`.
This note may contain old evaluation, spec, and engine concepts kept only as
historical context.

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
  - defines how observations and optional state become trading behavior
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
- `evaluation target`
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
