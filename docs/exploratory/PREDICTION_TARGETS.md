# Prediction Targets

Exploratory design note.

This file captures categories of targets the system may eventually predict.
It is not the current runtime contract, current CLI surface, or current trusted
mainline.

The list below is not exhaustive. It is a map of plausible target classes, not
a complete backlog or commitment.

Prefer:

- `README.md` for the current runtime path
- `OPERATING_BOUNDARIES.md` for the current operating posture
- `DESIGN.md` for the broader architecture
- `docs/portfolio-runtime-principles.md` for current portfolio / allocation terminology

What might eventually be worth predicting? This note focuses first on
market-facing targets, but predictive targets can also exist at the
hypothesis, allocation, and execution layers.

## Cross-sectional
- Which assets will outperform others over the next N days
- Sector rotation direction (tech → energy → defensives)
- Crypto pair spread convergence/divergence (e.g. ETH/BTC ratio mean-reversion)

## Directional
- Price direction over the next N hours/days
- Post-event price drift (FOMC, payrolls, earnings)
- Market overreaction — probability of rebound after sharp drops
- Seasonal patterns (month-end rebalancing, dividend dates, futures roll)

## Volatility & regime
- Volatility expansion vs contraction
- Regime transition timing (trend ↔ range)
- VIX term structure shifts (contango → backwardation)
- Tail risk probability — is an extreme move likely?

## Correlation & structure
- Correlation structure changes (normally uncorrelated assets becoming correlated)
- Cross-market lead-lag (US equities → Asian equities reaction)
- FX-equity linkage breakdown (carry trade unwind signals)

## Liquidity & microstructure
- Liquidity drying up (spread widening, thinning order book)
- Order flow imbalance (large buyer/seller detected)

## Sentiment & macro
- Macro indicator vs price divergence (rates up + stocks up → correction risk)
- Extreme sentiment readings (Fear & Greed → contrarian signal)
- Funding rate direction (crypto long/short bias → contrarian opportunity)

## On-chain (crypto-specific)
- Large exchange inflows (sell pressure signal)
- Miner/whale wallet movements

## Hypothesis / model health
- Which hypotheses are likely to retain predictive power over the next N cycles
- Which hypotheses are decaying due to crowding or regime change
- Which newly retained hypotheses are likely to become actionable live

## Allocation / portfolio construction
- Which sleeves should receive more capital over the next rebalance window
- Which hypotheses should be included or excluded from the current shortlist
- Which clusters are likely to be redundant despite strong standalone quality

## Execution / runtime quality
- Whether a signal is likely to persist long enough to trade
- Whether current market conditions justify trading now versus waiting
- Whether expected execution quality is too poor for a nominally valid signal
- Whether current data quality is sufficient to trust the runtime decision

---

## Future Implications If Promoted

- The DSL and feature catalog would need to cover inputs relevant to the chosen
  target classes
- Evaluation might need target-specific metrics rather than one shared metric
- Cross-sectional targets would require a different runtime contract from the
  current bounded per-sleeve directional flow
- Operational targets such as execution quality or hypothesis health would
  likely need different observation and scoring pipelines from market targets
