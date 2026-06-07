# Prediction Markets

This lane looks for event-market alpha candidates.

The first probe uses Polymarket public Gamma market data. It does not place
orders and does not require authentication.

## Commands

```bash
uv run python -m strategies.prediction_markets.current_polymarket_microstructure
```

## Current Status

This is not a trading strategy. It is a screen for finding event markets where
prediction modeling, market making, or information-flow work might be worth
building.
