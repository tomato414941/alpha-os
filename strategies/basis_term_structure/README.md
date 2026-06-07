# Basis Term Structure

This lane looks for dated futures basis dislocations.

It is separate from directional price prediction. The first screen uses public
Deribit BTC/ETH futures tickers and compares each dated future mark to the
reported index price.

## Commands

```bash
uv run python -m strategies.basis_term_structure.current_deribit_futures_basis
```

## Current Status

This is a current basis screen, not a trade instruction. Any candidate still
needs spot/perp hedge route, fees, margin, borrow/collateral cost, funding, and
execution-depth checks.
