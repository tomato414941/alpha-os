# Cash Rotation Results

Data:

```text
SPY QQQ TLT GLD
```

Fetch command:

```text
uv run python -m strategies.equity_index.fetch_market_data --symbols SPY QQQ TLT GLD
```

Backtest command:

```text
uv run python -m strategies.cash_rotation.backtest
```

Result:

```text
steps=482
total_return=0.367430
annualized_return=0.267406
annualized_volatility=0.240575
sharpe=1.105907
max_drawdown=-0.135577
mean_daily_turnover=0.043568
```

Interpretation:

This is the strongest of the new broad candidates. It is simple, has low
turnover, and has a much smaller drawdown than the current crypto momentum
candidates. It should be treated as a serious next strategy family, not just an
example.
