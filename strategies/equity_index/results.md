# Equity Index Results

Data:

```text
SPY QQQ IWM TLT GLD
```

Fetch command:

```text
uv run python -m strategies.equity_index.fetch_market_data
```

Backtest command:

```text
uv run python -m strategies.equity_index.backtest
```

Result:

```text
steps=482
total_return=0.243337
annualized_return=0.179311
annualized_volatility=0.281853
sharpe=0.727686
max_drawdown=-0.200477
mean_daily_turnover=0.155602
```

Interpretation:

The first ETF momentum rotation candidate is positive, but not obviously better
than a simple risk-on/risk-off candidate. It is useful as a non-crypto strategy
family for comparison.
