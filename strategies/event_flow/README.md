# Event Flow

This lane tests whether short-horizon order-flow information can become a
trading input.

It is deliberately separate from daily-bar momentum work. The first data path
uses Binance USD-M futures daily `aggTrades` and aggregates them into 5-minute
bars:

- close
- total traded quantity
- taker buy quantity
- taker sell quantity
- trade count

The second data path uses Binance USD-M futures daily `bookDepth` plus 1-minute
klines and derives:

- +/-1% liquidity notional imbalance
- +/-5% liquidity notional imbalance
- next 1-minute close return

Raw downloaded/aggregated market data is written under `market_data/`, which is
ignored by git.

## Commands

```bash
uv run python -m strategies.event_flow.fetch_aggtrade_sample
uv run python -m strategies.event_flow.flow_imbalance_screen
uv run python -m strategies.event_flow.fetch_book_depth_sample
uv run python -m strategies.event_flow.book_depth_imbalance_screen
```

## Current Status

The first screen is only a data-path and diagnostic check. It is not a
deployable strategy.

The current diagnostic asks whether 5-minute taker-flow imbalance predicts the
next 5-minute close-to-close return.

The book-depth diagnostic asks whether shallow/deep liquidity imbalance predicts
the next 1-minute close-to-close return.
