# Prediction Targets

Exploratory design note.

This file captures categories of targets the system may eventually predict.
It is not the current runtime contract, current CLI surface, or current trusted
mainline.

Prefer:

- `README.md` for the current runtime path
- `RECOVERY.md` for the current operating posture
- `DESIGN.md` for the broader architecture
- `docs/portfolio-runtime-principles.md` for current portfolio / allocation terminology

What can be profitably predicted? The more of these the system can predict,
the more diversified and stable the returns.

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

---

## Design implications

- The DSL and feature catalog should cover inputs relevant to all categories above
- Evaluation should reward prediction accuracy (IC) not just backtest returns (Sharpe)
- Cross-sectional predictions are market-direction-neutral and thus more robust
- Multiple weak predictions across categories compound into a strong edge
