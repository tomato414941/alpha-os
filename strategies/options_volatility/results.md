# Options Volatility Results

Run:

```bash
uv run python -m strategies.options_volatility.current_deribit_options_surface
uv run python -m strategies.options_volatility.current_deribit_options_realized_vol_labels
uv run python -m strategies.options_volatility.current_options_volatility_paper_tickets
```

This lane compresses public Deribit BTC/ETH option summaries into ATM IV, simple
5% OTM skew, adjacent-expiry term structure, and fast IV-vs-recent-realized
labels. It is a volatility-surface exploration probe, not a trade instruction.

Current paper candidates are written to:

- `current_deribit_options_surface.md`
- `current_deribit_options_realized_vol_labels.md`
- `current_options_volatility_paper_tickets.md`

The current stack includes BTC short-put-spread candidates and BTC/ETH calendar
spread watches. These still need actual option spread quotes, margin, tail-risk
limits, delta-hedge PnL, event timing, and realized-vol forecasts before any live
trade.
