# Trading Universe Design

## Problem

The system now evaluates 919 OHLCV assets for alpha exploration, but trading
is hardcoded to 3 assets (BTC, ETH, SOL). The evaluation universe (20 assets)
is selected for diversity of validation, not for profitability. There is no
mechanism to decide **which assets to trade** based on where alphas actually
have predictive power.

## Key Distinction

| Universe | Purpose | Selection criteria | Size |
|----------|---------|-------------------|------|
| price_signals | Exploration — evaluate expressions broadly | signal_type == "ohlcv" | ~919 |
| eval_universe | Validation — ensure alpha isn't overfit to one asset | Low pairwise correlation, long history | 20 |
| **trading_universe** | Profit — trade where we have edge | IC > 0, tradeable, liquid | Dynamic |

## Design

```
admitted alphas (each carries per-asset IC from cross-asset evaluation)
    |
    v
aggregate IC per asset (across all active alphas)
    |
    v
filter:
  - aggregate IC > threshold
  - venue available (Binance, Alpaca, etc.)
  - sufficient liquidity (volume, spread)
    |
    v
trading_universe (top N assets, recomputed periodically)
    |
    v
TC-weighted allocation across trading_universe
```

### What assets to trade is determined by alpha predictive power, not by humans.

## Prerequisites (Current Gaps)

1. **admission is BTC-fixed** — `--asset BTC` hardcoded, registry at `data/BTC/`.
   Needs to become asset-agnostic.
2. **per-asset IC not stored** — `evaluate_cross_asset` returns `{asset: ic}` but
   this is discarded after fitness averaging. Needs to be saved in alpha record.
3. **No trading_universe module** — Needs a new module that computes trading
   universe from aggregate per-asset IC + venue/liquidity filters.
4. **trader assumes static asset list** — `cross-trade --assets BTC,ETH,SOL`
   needs to accept dynamic universe from the trading_universe module.
5. **registry is asset-scoped** — `data/BTC/alpha_registry.db` assumes one
   asset. Cross-asset alphas need a single shared registry.

## Implementation Order

1. Save per-asset IC in alpha record (admission + registry schema change)
2. Make admission asset-agnostic (single shared registry)
3. Build trading_universe module (aggregate IC + filters)
4. Connect trader to dynamic trading_universe
5. Remove hardcoded `--assets` from trader service

## Constraints

- Venue availability: Binance (crypto), Alpaca (US stocks/ETFs), Paper (anything)
- Liquidity: minimum daily volume, maximum spread
- Memory: current server (7.6GB) limits concurrent data loading
- Start with paper trading on dynamic universe, graduate to real per venue
