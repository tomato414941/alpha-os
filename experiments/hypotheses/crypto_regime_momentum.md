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

Missing:

- A reproducible BTCUSDT/ETHUSDT data path with close, volume, funding rate,
  and open interest is not confirmed.
- The volume-confirmed trend operator is currently scoped to asset, equity, and
  ETF subject kinds, not crypto perpetuals.
- The exact candidate construction for combining momentum and regime inputs is
  not fixed.

Next smallest step:

- Confirm or create a reproducible BTCUSDT/ETHUSDT fixture with the required
  columns before creating a runtime manifest for this hypothesis.
