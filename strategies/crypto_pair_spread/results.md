# Crypto Pair Spread Results

Command:

```text
uv run python -m strategies.crypto_pair_spread.backtest --dataset-dir strategies/crypto/market_data/binance_spot_daily
```

Result:

```text
steps=826
total_return=-0.426867
annualized_return=-0.218056
annualized_volatility=0.224005
sharpe=-0.984258
max_drawdown=-0.455923
mean_daily_turnover=0.260291
```

Interpretation:

The first z-score spread candidate is weak. It does create a meaningfully
different strategy shape from long-only momentum, but this version is not a
live or paper candidate.
