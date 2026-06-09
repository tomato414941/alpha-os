# Current Modern Alpha Candidate Batch

This is a one-off modern-input candidate batch. It is not a queue, tracker,
framework, or reusable strategy registry.

Created at `2026-06-09T23:32Z`.

## Rule

Crypto directional candidates must have at least one non-price companion input:
liquidation/OI, order-flow, wallet/entity flow, attention/event state,
prediction-market odds, options/volatility, or cross-venue dislocation.

Rows that only have price, funding, and relative return are kept only as
`legacy_shallow_screen` and are not eligible for promotion.

Fixed evaluation:

- Primary horizon: `1h`.
- Secondary horizon: `4h`.
- Path check: `15m` adverse excursion only.
- Missing or stale input is recorded explicitly, not silently filled in.

## Batch Shape

| family | count | input freshness |
| --- | ---: | --- |
| liquidation + OI | 6 | fresh `/tmp` OKX liquidation snapshot |
| microstructure / order-flow | 6 | fresh `/tmp` Hyperliquid book/trade snapshot |
| wallet/entity flow | 4 | stale local seed-wallet output |
| attention/event state | 5 | fresh `/tmp` attention/price context |
| prediction-market odds | 4 | refreshed event-hedge candidates |
| options / vol surface | 3 | fresh `/tmp` Deribit option surface |
| cross-venue / basis / funding | 5 | stale local feasibility and basis rows |
| legacy shallow screens | 3 | old price/funding-only candidates |

Total rows: 36.

## Immediate Reads

- The strongest fresh modern inputs are liquidation/OI shocks and order-flow
  imbalance, not funding carry.
- ZEC is no longer allowed to be promoted as a funding/relative-strength story.
  It only remains interesting if the fresh liquidation/OI row survives.
- HYPE is conflicted: prediction/event and earlier relative weakness are not
  enough, and the current microstructure row is book/trade divergent.
- Wallet/entity rows are intentionally marked stale. They are useful as a source
  family, not as current tradable candidates.
- Cross-venue and basis rows are stale but retained because they are structurally
  modern. They require a fresh venue/hedge route before any promotion.
- Options rows are watch-only. They describe volatility/skew context, not naked
  spot direction.

## What Changed Versus The Previous Batch

- Funding-only candidates are demoted to `legacy_shallow_screen`.
- Every non-legacy crypto row now has a named companion input.
- The batch includes actual current order-flow, liquidation pressure,
  prediction odds, and options surface rows.
- Promotion blockers are explicit: stale input, depth/cost, mapping, hedge route,
  source quality, or conflicting inputs.

## Next Action

At the fixed `1h` horizon, score only rows whose input was fresh or whose stale
status was explicitly accepted as context. Do not promote stale wallet,
cross-venue, or basis rows until their source is refreshed.

Machine-readable rows are in
`strategies/current_modern_alpha_candidate_batch_20260609T2332Z.csv`.
