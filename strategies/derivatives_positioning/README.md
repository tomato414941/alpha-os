# Derivatives Positioning

This lane looks at multi-venue derivatives positioning: open interest, volume,
funding, basis, and spread.

The first current screen uses CoinGecko derivatives data. It is separate from
single-venue Hyperliquid/OKX perp maps and from historical Binance data.

## Commands

```bash
uv run python -m strategies.derivatives_positioning.current_coingecko_derivatives_positioning
```

## Current Status

This is a current positioning screen, not a trade instruction. Candidates still
need venue-specific depth, funding timing, margin, fees, borrow/hedge route, and
repeated forward labels.
