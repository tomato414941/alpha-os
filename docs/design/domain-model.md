# Domain Model

## Core Terms

| Term | Definition | Examples |
|------|-----------|---------|
| **feature** | Input data series from signal-noise. Raw or computed market observables that hypotheses consume. | `fear_greed`, `btc_ohlcv`, `vix_close`, `funding_rate_btc` |
| **hypothesis** | A single predictive logic that consumes features and produces predictions. A claim about market inefficiency; it may or may not have real predictive power. | `(sub fear_greed dxy)`, XGBoost model, RSI mean-reversion rule |
| **prediction** | The concrete output value a hypothesis produces for a given date and asset. This is what the pipeline evaluates. | `+0.3`, `-0.15` |
| **alpha** | Excess return over benchmark. It is an outcome, not a method. | Sharpe 0.5 after subtracting benchmark |

The primary data flow is:

`feature -> hypothesis -> prediction`

The evaluation pipeline scores predictions, not hypothesis internals.

## Strategy Hierarchy

alpha-os is centered on strongly predictive strategies.

```
strategy
├── strongly predictive
│   └── hypothesis
│       ├── DSL expression
│       ├── ML model
│       ├── classical indicator
│       └── ...
└── weakly predictive
    ├── arbitrage
    ├── market making
    └── ...
```

The platform may eventually hold multiple sleeves, but the current runtime is a
single predictive sleeve.

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

Hypotheses are evaluated by prediction quality. Sleeves are evaluated by
portfolio outcomes such as Sharpe, drawdown, and correlation.

## Hypothesis Dimensions

A hypothesis is defined by two orthogonal axes:

- **method**: how it is built
- **domain**: what market phenomenon it targets

### Method

| Method | What it does | Status |
|--------|-------------|--------|
| **DSL / GP** | Compose features via S-expression operators, evolved by genetic programming | Active |
| **Human-authored** | Fixed rules written by humans from domain knowledge | Active |
| **ML** | Learn patterns from features statistically | Planned |
| **LLM / NLP** | Extract predictions from unstructured text | Future |
| **Meta / ensemble** | Combine other hypotheses' predictions | Future |
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

- `hypothesis` for predictive units
- `prediction` for concrete outputs

Legacy names such as `alpha_id` remain only as compatibility residue and should
not become new source-of-truth concepts.
