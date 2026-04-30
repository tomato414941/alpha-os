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
