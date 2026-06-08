# Market Breadth

This lane scans broad crypto market data for volume and price dislocations.

It is not a trading strategy. It is a broad candidate-generation lane for
reversal, continuation, and chase-risk setups across many liquid crypto assets.
Forward labels use CoinGecko price history first and fall back to Hyperliquid
perp candles when CoinGecko cannot label a tradable symbol.

## Commands

```bash
uv run python -m strategies.market_breadth.current_volume_price_dislocation
uv run python -m strategies.market_breadth.current_volume_price_dislocation_history
uv run python -m strategies.market_breadth.current_volume_price_dislocation_labels
```
