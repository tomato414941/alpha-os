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

The latest tradable category labels are now partly mature. The sector momentum
direction mostly failed over the current short horizon.

Interpretation:

- `POL` is the only clear positive labeled row in the current sample.
- `ZEC`, `XPL`, `JUP`, `ONDO`, `TURBO`, and `PYTH` failed the current 15m
  sector-momentum direction.
- `HMSTR`, `CATI`, `LAB`, `H`, `ORBS`, `SAHARA`, and `HOME` are still pending
  or not directly usable as current promotion evidence.
- Category rotation remains useful as a broad context lane, but this refresh
  does not produce a deployable sector-continuation candidate.

## Current Category Perp Context

The perp context now deprioritizes most mature rows. `POL` worked directionally,
but funding support was weak, so it is not promoted.

Interpretation:

- `ZEC` has funding support but failed the sector direction label.
- `XPL`, `JUP`, `ONDO`, `TURBO`, `PYTH`, `XMR`, and `LINK` also failed the
  current sector direction label.
- `POL` is positive on price, but weak funding support keeps it as
  `deprioritize`.
- The lane should stay in exploration until repeated category labels beat
  costs and show a clearer link between category momentum and tradable perps.
