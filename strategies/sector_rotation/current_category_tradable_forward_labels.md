# Current Category Tradable Forward Labels

This maps CoinGecko category rotation into Hyperliquid-tradable constituents and labels subsequent 15m/1h returns. It is a sector rotation label, not a trade instruction.

| category | coin | action | change24 | dir | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DRC-20 | DOGI | sector_momentum_watch | 125.8484 | 1 |  |  |  |  | not_hyperliquid |
| DRC-20 | FIWB | sector_momentum_watch | 125.8484 | 1 |  |  |  |  | not_hyperliquid |
| DRC-20 | DALL | sector_momentum_watch | 125.8484 | 1 |  |  |  |  | not_hyperliquid |
| Bridged Stablecoin | BSC-USD | sector_stress_watch | -99.8534 | -1 |  |  |  |  | not_hyperliquid |
| Bridged Stablecoin | USDT0 | sector_stress_watch | -99.8534 | -1 |  |  |  |  | not_hyperliquid |
| Arcade Games | BEAT | sector_momentum_watch | 91.0821 | 1 |  |  |  |  | not_hyperliquid |
| Arcade Games | HMSTR | sector_momentum_watch | 91.0821 | 1 |  |  |  |  | tradable_pending_label |
| Arcade Games | PEPECOIN | sector_momentum_watch | 91.0821 | 1 |  |  |  |  | not_hyperliquid |
| Bridged Stablecoin | USDC | sector_stress_watch | -99.8534 | -1 |  |  |  |  | not_hyperliquid |
| Telegram Apps | BEAT | sector_momentum_watch | 60.7626 | 1 |  |  |  |  | not_hyperliquid |
| Telegram Apps | FLOKI | sector_momentum_watch | 60.7626 | 1 |  |  |  |  | not_hyperliquid |
| Telegram Apps | CATI | sector_momentum_watch | 60.7626 | 1 |  |  |  |  | tradable_pending_label |
| OpenServ Ecosystem | ROUTER | sector_momentum_watch | 44.4051 | 1 |  |  |  |  | not_hyperliquid |
| OpenServ Ecosystem | BETTER | sector_momentum_watch | 44.4051 | 1 |  |  |  |  | not_hyperliquid |
| OpenServ Ecosystem | COBOT | sector_momentum_watch | 44.4051 | 1 |  |  |  |  | not_hyperliquid |
| ERC 404 | PANDORA | sector_momentum_watch | 36.7653 | 1 |  |  |  |  | tradable_pending_label |
| ERC 404 | DEFROGS | sector_momentum_watch | 36.7653 | 1 |  |  |  |  | not_hyperliquid |
| ERC 404 | PURSE | sector_momentum_watch | 36.7653 | 1 |  |  |  |  | not_hyperliquid |
| Privacy | ZEC | sector_momentum_watch | 25.9565 | 1 |  |  |  |  | tradable_pending_label |
| Privacy | XMR | sector_momentum_watch | 25.9565 | 1 |  |  |  |  | tradable_pending_label |
| Privacy | LINK | sector_momentum_watch | 25.9565 | 1 |  |  |  |  | tradable_pending_label |
| Groypad Ecosystem | FAST | sector_momentum_watch | 30.8687 | 1 |  |  |  |  | not_hyperliquid |
| Groypad Ecosystem | LNG | sector_momentum_watch | 30.8687 | 1 |  |  |  |  | not_hyperliquid |
| Groypad Ecosystem | PYONYA | sector_momentum_watch | 30.8687 | 1 |  |  |  |  | not_hyperliquid |
| NFT Lending/Borrowing | AQT | sector_stress_watch | -27.1410 | -1 |  |  |  |  | not_hyperliquid |

## Interpretation

This is still one snapshot. It does not model constituent weighting, category membership quality, liquidity, costs, or repeated evidence. Rows with `not_hyperliquid` are useful only as context because they are not directly tradable through this venue.
