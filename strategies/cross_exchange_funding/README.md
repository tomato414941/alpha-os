# Cross-Exchange Funding

This lane looks for funding-rate spreads across venues.

The first screen uses Hyperliquid's public `predictedFundings` response, which
includes venue-level predicted funding for assets across venues such as
Hyperliquid, Binance perpetuals, and Bybit perpetuals.

The screen is not a trade recommendation. It does not yet verify:

- whether both venues are accessible to the same account
- maker/taker fees and spread
- borrow and margin constraints
- open-interest caps
- position limits
- transfer and collateral constraints
- order book depth
- whether the predicted funding persists until execution

## Commands

```bash
uv run python -m strategies.cross_exchange_funding.venue_access_probe
uv run python -m strategies.cross_exchange_funding.current_funding_spread
uv run python -m strategies.cross_exchange_funding.current_funding_feasibility
uv run python -m strategies.cross_exchange_funding.current_okx_hl_funding_spread
uv run python -m strategies.cross_exchange_funding.okx_hl_funding_persistence_probe
uv run python -m strategies.cross_exchange_funding.okx_hl_paper_ticket
uv run python -m strategies.cross_exchange_funding.okx_hl_order_constraints
uv run python -m strategies.cross_exchange_funding.okx_hl_funding_alignment
uv run python -m strategies.cross_exchange_funding.okx_hl_fee_sensitivity
uv run python -m strategies.cross_exchange_funding.okx_hl_book_depth
uv run python -m strategies.cross_exchange_funding.okx_hl_candidate_score
uv run python -m strategies.cross_exchange_funding.okx_hl_execution_cost_score
uv run python -m strategies.cross_exchange_funding.okx_hl_candidate_triage
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_score
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_triage
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_monitor
```

Focused OKX-Hyperliquid monitor:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_funding_persistence_probe --samples 12 --delay-seconds 10 --assets BTC JTO BABY ZEC --output-path strategies/cross_exchange_funding/okx_hl_funding_persistence_focus.csv --summary-output-path strategies/cross_exchange_funding/okx_hl_funding_persistence_focus_summary.csv
uv run python -m strategies.cross_exchange_funding.okx_hl_candidate_score --summary-path strategies/cross_exchange_funding/okx_hl_funding_persistence_focus_summary.csv --output-path strategies/cross_exchange_funding/okx_hl_candidate_score_focus.csv --md-output-path strategies/cross_exchange_funding/okx_hl_candidate_score_focus.md
uv run python -m strategies.cross_exchange_funding.okx_hl_execution_cost_score --summary-path strategies/cross_exchange_funding/okx_hl_funding_persistence_focus_summary.csv --assets BTC JTO ZEC BABY
uv run python -m strategies.cross_exchange_funding.okx_hl_candidate_triage
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_score
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_triage
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_monitor --samples 6 --delay-seconds 10
```
