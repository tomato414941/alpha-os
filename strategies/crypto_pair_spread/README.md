# Crypto Pair Spread

Relative-value crypto strategy candidate.

Hypothesis:

When a liquid crypto pair ratio is far from its recent mean, a market-neutral
spread position can capture mean reversion with lower directional market
exposure than long-only momentum.

Current shape:

- input: daily close history
- output: target weights
- candidate: z-score pair spread
- pairs: `BTCUSDT/ETHUSDT`, `SOLUSDT/ETHUSDT`, `DOGEUSDT/BTCUSDT`

Run:

```text
uv run python -m strategies.crypto_pair_spread.backtest --dataset-dir strategies/crypto/market_data/binance_spot_daily
```
