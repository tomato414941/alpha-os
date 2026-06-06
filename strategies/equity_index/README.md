# Equity Index

ETF rotation strategy candidates.

Hypothesis:

Rotating among broad equity/risk assets by medium-term trend can provide a
simple non-crypto strategy family for comparison against crypto candidates.

Current universe:

```text
SPY QQQ IWM TLT GLD
```

Fetch data:

```text
uv run python -m strategies.equity_index.fetch_market_data
```

Run:

```text
uv run python -m strategies.equity_index.backtest
```
