# Prediction Markets

This lane looks for event-market alpha candidates.

The first probe uses Polymarket public Gamma market data. It does not place
orders and does not require authentication.

## Commands

```bash
uv run python -m strategies.prediction_markets.current_polymarket_microstructure
uv run python -m strategies.prediction_markets.current_polymarket_microstructure_monitor
uv run python -m strategies.prediction_markets.current_polymarket_clob_depth
uv run python -m strategies.prediction_markets.current_event_news_pressure
uv run python -m strategies.prediction_markets.current_event_probability_gap
uv run python -m strategies.prediction_markets.current_event_probability_paper_tickets
```

## Current Status

This is not a trading strategy. It is a screen for finding event markets where
prediction modeling, market making, or information-flow work might be worth
building.
