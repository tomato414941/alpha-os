# Crypto Market Structure

This lane tests whether crypto-specific market structure is more useful than
daily close momentum alone.

It starts with Binance USD-M futures public data:

- perp daily klines
- funding rate history
- premium index daily klines
- taker buy volume embedded in futures klines

Open interest, taker buy/sell ratio, and basis REST endpoints are useful
short-term diagnostics, but Binance currently limits those historical endpoints
to the latest 30 days. They are not the first long-horizon backtest lane.

## Current Question

Can funding crowding, premium dislocation, taker flow imbalance, or volume
expansion explain next-day crypto returns better than close-only indicators?

## Commands

```bash
uv run python -m strategies.crypto_market_structure.fetch_market_data --start-date 2024-01-01
uv run python -m strategies.crypto_market_structure.diagnostics
uv run python -m strategies.crypto_market_structure.predictive_screen
uv run python -m strategies.crypto_market_structure.predictive_exposure_audit
uv run python -m strategies.crypto_market_structure.broad_model_screen
uv run python -m strategies.crypto_market_structure.funding_carry
uv run python -m strategies.crypto_market_structure.funding_carry_cost_stress
```
