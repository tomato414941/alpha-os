# Options Volatility

This lane looks for option-surface dislocations, starting with Deribit BTC/ETH
public option summaries.

It is not a volatility strategy yet. The first probe only extracts ATM IV,
simple 5% OTM skew, and term structure candidates.

## Commands

```bash
uv run python -m strategies.options_volatility.current_deribit_options_surface
```
