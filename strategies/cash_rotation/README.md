# Cash Rotation

Risk-on/risk-off strategy candidate.

Hypothesis:

When the broad equity regime is positive, hold a risk asset. When it is not,
rotate into the strongest defensive asset or cash.

Current symbols:

```text
SPY QQQ TLT GLD
```

This candidate currently reuses the equity index Yahoo daily data directory.

Fetch data:

```text
uv run python -m strategies.equity_index.fetch_market_data --symbols SPY QQQ TLT GLD
```

Run:

```text
uv run python -m strategies.cash_rotation.backtest
```
