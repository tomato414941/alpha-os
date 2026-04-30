# Crypto Regime Momentum

Hypothesis UID: `hyp_5880aba5`

## Claim

Adding market regime information to simple BTC/ETH momentum improves out-of-
sample adoption decisions versus the baseline.

## Market

Crypto perpetual futures.

## Universe

- BTCUSDT
- ETHUSDT

## Target

Next 1 day return.

## Inputs

- 7 day return
- 30 day return
- realized volatility
- volume change
- funding rate
- open interest

## Baseline

BTC/ETH simple momentum.

## Decision Rule

Promote only if the candidate beats the baseline after costs without
unacceptable drawdown or turnover degradation.

## Rejection Rule

Reject if the edge disappears after costs, reverses under small date changes,
or requires unavailable data.

## Feasibility Check

Confirmed:

- The evaluation path can build features from `close`, `volume`,
  `funding_rate`, and `open_interest` columns when an observation frame provides
  them.
- The checked-in fixture path currently provides `close` only.
- The current environment does not have the optional `signal-noise` package
  installed.
- Existing signal operators include time-series trend, funding carry, realized
  volatility regime variants, and volume-confirmed trend.

Still unresolved:

- The current dataset has close, volume, funding rate, and open interest.
- The funding and open interest daily aggregation definitions are suitable for
  feature experimentation, not final execution accounting.
- The volume-confirmed trend operator is currently scoped to asset, equity, and
  ETF subject kinds, not crypto perpetuals.
- The exact candidate construction for combining momentum and regime inputs is
  not fixed.

Next smallest step:

- Fix the first candidate construction before creating a runtime manifest.

Evidence data:

- `experiments/datasets/ds_crypto_btc_eth_daily_2024_2025/`
