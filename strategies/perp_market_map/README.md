# Perp Market Map

This lane maps perpetual futures markets before turning anything into a
strategy.

The first probe uses Hyperliquid public market contexts:

- current funding
- open interest
- 24h notional volume
- premium
- mark/oracle dislocation
- impact spread

## Commands

```bash
uv run python -m strategies.perp_market_map.current_hyperliquid_snapshot
uv run python -m strategies.perp_market_map.current_hyperliquid_dislocation_candidates
uv run python -m strategies.perp_market_map.current_hyperliquid_dislocation_forward_labels
uv run python -m strategies.perp_market_map.current_hyperliquid_dislocation_execution_check
uv run python -m strategies.perp_market_map.current_crowding_reversion_screen
uv run python -m strategies.perp_market_map.current_crowding_reversion_monitor
uv run python -m strategies.candidate_validation.current_hl_signal_forward_labels
uv run python -m strategies.perp_market_map.current_crowding_reversion_validated_candidates
uv run python -m strategies.perp_market_map.current_crowding_reversion_execution_check
uv run python -m strategies.perp_market_map.current_crowding_reversion_paper_outcome
uv run python -m strategies.perp_market_map.current_okx_perp_pressure
uv run python -m strategies.perp_market_map.current_okx_perp_pressure_forward_labels
```

## Current Status

This is not a trading strategy. It is a market map for finding where carry,
crowding, dislocation, and liquidity might justify deeper work. The
crowding/reversion screen only proposes watch candidates; it does not prove
future returns, fills, or liquidation pressure.
