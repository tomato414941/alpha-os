# Derivatives Positioning Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.derivatives_positioning.current_coingecko_derivatives_positioning
```

Interpretation:

- High open interest with material funding can indicate crowded positioning.
- Basis plus funding dislocation can point to relative-value or reversion
  candidates.
- Large 24h moves with meaningful OI can be continuation or reversal risk.
- Any candidate still needs venue-specific depth, funding timing, fees, margin,
  and forward labels.

## Current Candidates

- `WhiteBIT Futures ZEC_PERP`: high OI/volume with material funding.
- `SOL-USDT` / `SOLUSDTM` across CoinUp and KuCoin: OI/funding crowding watch.
- `TRXUSDT` across Bybit, CoinW, and Binance: repeated OI/funding crowding
  context.
- `Bybit ZECUSDT`: basis plus funding dislocation watch.

Important caveat:

- CoinGecko derivatives data is aggregated. Before paper action, check the
  actual venue book, funding timestamp, fees, margin, and whether the contract
  is accessible.
