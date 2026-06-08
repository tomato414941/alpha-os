# Current Category Tradable Forward Labels

This maps CoinGecko category rotation into Hyperliquid-tradable constituents and labels subsequent 15m/1h returns. It is a sector rotation label, not a trade instruction.

| category | coin | action | change24 | dir | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Privacy | ZEC | sector_momentum_watch | 26.1004 | 1 | 0.000249 | 0.000249 |  |  | tradable_labeled |
| Quantum-Resistant | ZEC | sector_momentum_watch | 19.0059 | 1 | 0.000249 | 0.000249 |  |  | tradable_labeled |
| Zero Knowledge (ZK) | ZEC | sector_momentum_watch | 15.2767 | 1 | 0.000249 | 0.000249 |  |  | tradable_labeled |
| Privacy Coins | ZEC | sector_momentum_watch | 14.3203 | 1 | 0.000249 | 0.000249 |  |  | tradable_labeled |
| DRC-20 | DOGI | sector_momentum_watch | 125.7495 | 1 |  |  |  |  | not_hyperliquid |
| DRC-20 | FIWB | sector_momentum_watch | 125.7495 | 1 |  |  |  |  | not_hyperliquid |
| DRC-20 | DALL | sector_momentum_watch | 125.7495 | 1 |  |  |  |  | not_hyperliquid |
| Bridged Stablecoin | BSC-USD | sector_stress_watch | -99.8533 | -1 |  |  |  |  | not_hyperliquid |
| Bridged Stablecoin | USDT0 | sector_stress_watch | -99.8533 | -1 |  |  |  |  | not_hyperliquid |
| Arcade Games | BEAT | sector_momentum_watch | 91.6029 | 1 |  |  |  |  | not_hyperliquid |
| Arcade Games | HMSTR | sector_momentum_watch | 91.6029 | 1 |  |  |  |  | tradable_pending_label |
| Arcade Games | PEPECOIN | sector_momentum_watch | 91.6029 | 1 |  |  |  |  | not_hyperliquid |
| Bridged Stablecoin | USDC | sector_stress_watch | -99.8533 | -1 |  |  |  |  | not_hyperliquid |
| Telegram Apps | BEAT | sector_momentum_watch | 60.6694 | 1 |  |  |  |  | not_hyperliquid |
| Telegram Apps | FLOKI | sector_momentum_watch | 60.6694 | 1 |  |  |  |  | not_hyperliquid |
| Telegram Apps | CATI | sector_momentum_watch | 60.6694 | 1 |  |  |  |  | tradable_pending_label |
| OpenServ Ecosystem | ROUTER | sector_momentum_watch | 48.7109 | 1 |  |  |  |  | not_hyperliquid |
| OpenServ Ecosystem | BETTER | sector_momentum_watch | 48.7109 | 1 |  |  |  |  | not_hyperliquid |
| OpenServ Ecosystem | COBOT | sector_momentum_watch | 48.7109 | 1 |  |  |  |  | not_hyperliquid |
| ERC 404 | PANDORA | sector_momentum_watch | 36.8360 | 1 |  |  |  |  | tradable_pending_label |
| ERC 404 | DEFROGS | sector_momentum_watch | 36.8360 | 1 |  |  |  |  | not_hyperliquid |
| ERC 404 | PURSE | sector_momentum_watch | 36.8360 | 1 |  |  |  |  | not_hyperliquid |
| Privacy | XMR | sector_momentum_watch | 26.1004 | 1 |  |  |  |  | tradable_pending_label |
| Privacy | LINK | sector_momentum_watch | 26.1004 | 1 |  |  |  |  | tradable_pending_label |
| Groypad Ecosystem | FAST | sector_momentum_watch | 31.8387 | 1 |  |  |  |  | not_hyperliquid |

## Interpretation

This is still one snapshot. It does not model constituent weighting, category membership quality, liquidity, costs, or repeated evidence. Rows with `not_hyperliquid` are useful only as context because they are not directly tradable through this venue.
