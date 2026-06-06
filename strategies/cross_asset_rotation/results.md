# Cross-Asset Rotation Results

Data preparation:

```text
uv run python -m strategies.cross_asset_rotation.prepare_market_data
```

Backtest:

```text
uv run python -m strategies.cross_asset_rotation.backtest
```

Result:

```text
variant=top_momentum_126_252
steps=356
total_return=0.057098
annualized_return=0.058583
annualized_volatility=0.540308
sharpe=0.375915
max_drawdown=-0.341379
mean_daily_turnover=0.058989

variant=vol_adjusted_momentum_126_252
steps=356
total_return=0.125395
annualized_return=0.128761
annualized_volatility=0.384964
sharpe=0.509279
max_drawdown=-0.257912
mean_daily_turnover=0.171348

variant=risk_on_off_126
steps=482
total_return=-0.045339
annualized_return=-0.034526
annualized_volatility=0.553639
sharpe=0.212353
max_drawdown=-0.521129
mean_daily_turnover=0.118257
```

Interpretation:

The first cross-asset candidates are weak. Adding crypto to the ETF rotation
universe does not help in this simple daily-close form. Vol-adjusted momentum
is the least bad variant, but it does not look like a current lead candidate.
