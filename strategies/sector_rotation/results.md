# Sector Rotation Results

Run:

```bash
uv run python -m strategies.sector_rotation.current_coingecko_category_rotation
uv run python -m strategies.sector_rotation.current_category_tradable_forward_labels
```

This lane ranks public CoinGecko crypto categories by 24h market-cap change,
scale, and volume. It is a sector-rotation context probe, not a trade
instruction.

## Current CoinGecko Category Rotation

| category | 24h change | market cap | volume 24h | top coins | action | score |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Arcade Games | 67.7407 | 890977441 | 54949085 | audiera;hamster-kombat;pepecoin-2 | sector_momentum_watch | 1130.5802 |
| Telegram Apps | 46.3304 | 1269976123 | 107247578 | audiera;floki;catizen | sector_momentum_watch | 793.8335 |
| Four.meme Ecosystem (BNB Memes) | 41.3405 | 1904783518 | 646269097 | siren-2;bianrensheng;hakimi | sector_momentum_watch | 747.8612 |
| Echo Launchpad | 38.9820 | 4445890370 | 166637511 | lab;plasma | sector_momentum_watch | 696.5972 |
| DRC-20 | -65.7620 | 2371121 | 49 | dogi;fiwb-doginals;dall-doginals | sector_stress_watch | 530.8842 |
| Analytics | 29.0012 | 5652481061 | 240650068 | lab;pyth-network;the-graph | sector_momentum_watch | 525.8977 |
| Sticker-Themed Coins | 34.1750 | 21372376 | 592375 | utya;paper-plane;cubigator | sector_momentum_watch | 447.7757 |
| Launchpad | 22.9892 | 6966601414 | 360253113 | lab;pump-fun;jupiter-exchange-solana | sector_momentum_watch | 422.9936 |
| Farcaster Ecosystem | 25.8666 | 73004819 | 13034841 | degen-base;doginme;the-doge-nft | sector_momentum_watch | 387.4418 |
| Privacy | 19.2986 | 29239306134 | 3219053958 | zcash;monero;chainlink | sector_momentum_watch | 385.4650 |

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
| Echo Launchpad | XPL | sector_momentum_watch | 38.9820 | 1 | 0.003547 | 0.003547 |  |  | tradable_labeled |
| AI Meme | TURBO | sector_momentum_watch | 10.5854 | 1 | 0.003480 | 0.003480 |  |  | tradable_labeled |
| Binance Alpha Spotlight | ONDO | sector_momentum_watch | 15.0026 | 1 | 0.002897 | 0.002897 |  |  | tradable_labeled |
| Launchpad | PUMP | sector_momentum_watch | 22.9892 | 1 | 0.002658 | 0.002658 |  |  | tradable_labeled |
| AI Meme | FARTCOIN | sector_momentum_watch | 10.5854 | 1 | 0.002312 | 0.002312 |  |  | tradable_labeled |
| Analytics | PYTH | sector_momentum_watch | 29.0012 | 1 | 0.001974 | 0.001974 |  |  | tradable_labeled |
| Privacy | LINK | sector_momentum_watch | 19.2986 | 1 | 0.001812 | 0.001812 |  |  | tradable_labeled |
| Zero Knowledge (ZK) | POL | sector_momentum_watch | 8.9346 | 1 | 0.001584 | 0.001584 |  |  | tradable_labeled |
| Launchpad | JUP | sector_momentum_watch | 22.9892 | 1 | 0.001556 | 0.001556 |  |  | tradable_labeled |
| Privacy | XMR | sector_momentum_watch | 19.2986 | 1 | -0.000478 | -0.000478 |  |  | tradable_labeled |

Interpretation:

- Category leaders are often not directly tradable on Hyperliquid, so category
  rotation needs a tradability map before it can become useful.
- The first tradable continuation labels are positive for `XPL`, `TURBO`,
  `ONDO`, `PUMP`, `FARTCOIN`, `PYTH`, `LINK`, `POL`, and `JUP`.
- This is still a single snapshot. It needs repeated samples, constituent
  weighting, liquidity, costs, and category-family reversal/continuation tests.
