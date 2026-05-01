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

For each asset:

- Compute 7 day close-to-close return.
- Hold long if the 7 day return is positive.
- Hold flat otherwise.

Portfolio construction:

- Equal weight across active assets.
- If both assets are flat, hold cash.
- Rebalance daily.

This baseline is intentionally simple. It asks whether plain BTC/ETH momentum
already captures the available edge before adding regime inputs.

## First Candidate

For each asset:

- Start from the same 7 day momentum rule as the baseline.
- Require 30 day return to be positive.
- Do not enter when funding rate is positive and above its trailing 60 day
  median.
- Cut position size in half when 20 day realized volatility is above its
  trailing 60 day median.
- Cut position size in half when 7 day open interest growth is positive while
  7 day return is negative.

Portfolio construction:

- Equal weight across candidate asset signals after position scaling.
- If both assets are flat, hold cash.
- Rebalance daily.

The candidate is deliberately not optimized. It tests one claim: whether simple
momentum improves when crowded or stressed regimes are avoided.

## First Evaluation

## Decision Rule

Promote only if the candidate beats the baseline after costs without
unacceptable drawdown or turnover degradation.

Primary comparison:

- Candidate mean daily net return must be higher than baseline.
- Candidate max drawdown must not be worse than baseline by more than 10%.
- Candidate turnover must not exceed 2x baseline turnover.
- Candidate must not win only in one asset while losing badly in the other.

Initial period:

- Train/lookback warmup: 2024-01-01 through 2024-03-31.
- Evaluation: 2024-04-01 through 2025-12-31.

Data source:

- `experiments/datasets/ds_crypto_btc_eth_daily_2024_2025/`

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
- The current dataset has close, volume, funding rate, and open interest.
- The first baseline, candidate, evaluation period, and comparison rule are
  fixed.

Still unresolved:

- The funding and open interest daily aggregation definitions are suitable for
  feature experimentation, not final execution accounting.
- The volume-confirmed trend operator is currently scoped to asset, equity, and
  ETF subject kinds, not crypto perpetuals.
- The first candidate has not yet been implemented and compared against the
  baseline.

Next smallest step:

- Implement the first baseline and candidate comparison directly against the
  checked-in dataset before creating a runtime manifest.

Evidence data:

- `experiments/datasets/ds_crypto_btc_eth_daily_2024_2025/`
