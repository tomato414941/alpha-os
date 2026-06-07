# Sector Rotation Results

Run:

```bash
uv run python -m strategies.sector_rotation.current_coingecko_category_rotation
uv run python -m strategies.sector_rotation.current_category_tradable_forward_labels
uv run python -m strategies.sector_rotation.current_category_perp_context
```

This lane ranks public CoinGecko crypto categories by 24h market-cap change,
scale, and volume. It is a sector-rotation context probe, not a trade
instruction.

## Current CoinGecko Category Rotation

| category | 24h change | market cap | volume 24h | top coins | action | score |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Arcade Games | 73.9737 | 924084749 | 57617816 | audiera;hamster-kombat;pepecoin-2 | sector_momentum_watch | 1237.3032 |
| Telegram Apps | 49.1236 | 1294217562 | 108897985 | audiera;floki;catizen | sector_momentum_watch | 842.4214 |
| Four.meme Ecosystem (BNB Memes) | 35.2284 | 1822413266 | 638149487 | siren-2;bianrensheng;hakimi | sector_momentum_watch | 636.4216 |
| Echo Launchpad | 33.9457 | 4284785715 | 161673893 | lab;plasma | sector_momentum_watch | 605.6107 |
| DRC-20 | -65.3981 | 2396322 | 49 | dogi;fiwb-doginals;dall-doginals | sector_stress_watch | 528.1544 |
| Sticker-Themed Coins | 36.1115 | 21680848 | 602888 | utya;paper-plane;cubigator | sector_momentum_watch | 473.6502 |
| Analytics | 25.3886 | 5494185444 | 236924014 | lab;pyth-network;the-graph | sector_momentum_watch | 459.9023 |
| Farcaster Ecosystem | 26.7476 | 73515807 | 13014510 | degen-base;doginme;the-doge-nft | sector_momentum_watch | 400.7005 |
| Privacy | 19.2678 | 29231761053 | 3157200050 | zcash;monero;chainlink | sector_momentum_watch | 384.6856 |
| Launchpad | 20.6313 | 6833039111 | 357259361 | lab;pump-fun;jupiter-exchange-solana | sector_momentum_watch | 379.3605 |

Interpretation:

- Arcade Games, Telegram Apps, BNB meme, launchpad, analytics, and privacy
  categories show strong current rotation.
- Several top categories are likely thin or hard to trade directly, so this is
  not deployable until constituent liquidity is checked.
- The next step is mapping rotating categories to tradable symbols and labeling
  category continuation/reversal.

## Current Category Tradable Forward Labels

| category | coin | action | change24 | dir | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Arcade Games | HMSTR | sector_momentum_watch | 73.9737 | 1 |  |  |  |  | tradable_pending_label |
| Telegram Apps | CATI | sector_momentum_watch | 49.1236 | 1 |  |  |  |  | tradable_pending_label |
| Echo Launchpad | XPL | sector_momentum_watch | 33.9457 | 1 |  |  |  |  | tradable_pending_label |
| Privacy | ZEC | sector_momentum_watch | 19.2678 | 1 |  |  |  |  | tradable_pending_label |
| Launchpad | JUP | sector_momentum_watch | 20.6313 | 1 |  |  |  |  | tradable_pending_label |
| Binance Alpha Spotlight | ONDO | sector_momentum_watch | 13.7373 | 1 |  |  |  |  | tradable_pending_label |
| Analytics | PYTH | sector_momentum_watch | 25.3886 | 1 |  |  |  |  | tradable_pending_label |
| Privacy | LINK | sector_momentum_watch | 19.2678 | 1 |  |  |  |  | tradable_pending_label |
| AI Meme | FARTCOIN | sector_momentum_watch | 9.5359 | 1 |  |  |  |  | tradable_pending_label |
| Four.meme Ecosystem (BNB Memes) | SIREN | sector_momentum_watch | 35.2284 | 1 |  |  |  |  | not_hyperliquid |

Interpretation:

- Category leaders are often not directly tradable on Hyperliquid, so category
  rotation needs a tradability map before it can become useful.
- The current tradable category labels are mostly pending because this is a fresh
  refresh.
- `HMSTR`, `CATI`, `XPL`, `ZEC`, `JUP`, and `ONDO` are the immediate label queue.
- The previous positive labeled snapshot is no longer treated as current
  evidence; it should only be used as historical context until repeated.

## Current Category Perp Context

| category | symbol | dir | dir15 | funding support | HL funding | OKX funding | score | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Privacy | ZEC | 1 |  | 0.75674928 | -0.33144949 | -0.75674928 | 1.135357 | wait_for_label | sector label is not mature yet |
| Zero Knowledge (ZK) | ZEC | 1 |  | 0.75674928 | -0.33144949 | -0.75674928 | 0.926691 | wait_for_label | sector label is not mature yet |
| Echo Launchpad | XPL | 1 |  | 0.02772491 | 0.10950000 | -0.02772491 | 0.706639 | wait_for_label | sector label is not mature yet |
| Launchpad | JUP | 1 |  | 0.03652307 | -0.03652307 |  | 0.449150 | wait_for_label | sector label is not mature yet |
| Arcade Games | HMSTR | 1 |  | -0.10950000 | 0.10950000 |  | 0.369973 | wait_for_label | sector label is not mature yet |
| Binance Alpha Spotlight | ONDO | 1 |  | 0.06860749 | 0.10950000 | -0.06860749 | 0.343415 | wait_for_label | sector label is not mature yet |
| Echo Launchpad | LAB | 1 |  | -0.05475000 |  | 0.05475000 | 0.124164 | wait_for_label | sector label is not mature yet |
| AI Meme | TURBO | 1 |  | 0.39823748 | -0.39823748 |  | 0.116261 | wait_for_label | sector label is not mature yet |

Interpretation:

- This refresh created a sector-perp waiting queue, not deployable candidates.
- Non-perp category constituents are excluded from this context screen.
- `ZEC`, `XPL`, `JUP`, `ONDO`, and `TURBO` have some funding support for the
  sector direction; `HMSTR` is tradable but funding is against the long
  direction, so it is no longer the top sector-perp context candidate.
- The next useful step is to rerun labels after the 15m horizon matures and
  separate category momentum from perp-carry support.
