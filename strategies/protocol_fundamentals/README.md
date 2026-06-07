# Protocol Fundamentals

This lane uses public protocol fee data as a non-price alpha source.

The research question is whether protocol fee growth or revenue concentration
can become a useful input for tradable tokens such as HYPE, AAVE, UNI, ENA,
JUP, AERO, and CRV.

## Commands

```bash
uv run python -m strategies.protocol_fundamentals.current_protocol_fee_screen
uv run python -m strategies.protocol_fundamentals.current_protocol_fee_valuation
uv run python -m strategies.protocol_fundamentals.current_protocol_fee_candidate_review
uv run python -m strategies.protocol_fundamentals.current_protocol_fee_price_context
uv run python -m strategies.protocol_fundamentals.current_protocol_fee_price_lag_history
uv run python -m strategies.protocol_fundamentals.current_protocol_fee_price_lag_labels
```
