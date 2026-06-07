# Sector Rotation Results

Run:

```bash
uv run python -m strategies.sector_rotation.current_coingecko_category_rotation
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
