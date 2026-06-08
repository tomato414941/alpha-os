# Current Hyperliquid Seed Wallet Flow

This turns a small public seed-wallet set into wallet-flow observations. It is not a copy-trading rule and the seed wallets are not verified entities.

| wallet | coin | action | fills | net buy USD | closed PnL | fees | net PnL | position | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| public_hypertracker_example | xyz:CL | watch_recent_wallet_sell_flow | 324 | -708.12 | 708.29 | 19.53 | 688.76 | 0.000000 | 70.71 | label public_hypertracker_example/xyz:CL wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | ETH | watch_wallet_long_pressure | 79 | 6903.69 | 436.91 | 22.33 | 414.58 | 2.796600 | 60.98 | label public_hypertracker_example/ETH wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | ZEC | watch_wallet_long_pressure | 40 | 49.59 | 449.71 | 5.95 | 443.76 | 1.320000 | 49.03 | label public_hypertracker_example/ZEC wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | hyna:ETH | watch_recent_wallet_sell_flow | 40 | -385.04 | 385.01 | 15.67 | 369.33 | 0.000000 | 41.32 | label public_hypertracker_example/hyna:ETH wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | hyna:BTC | watch_recent_wallet_sell_flow | 40 | -281.87 | 281.86 | 12.30 | 269.56 | 0.000000 | 31.24 | label public_hypertracker_example/hyna:BTC wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_live_bot_example | XRP | watch_wallet_long_pressure | 201 | 8.13 | 19.35 | 2.21 | 17.13 | 24.000000 | 21.75 | label public_live_bot_example/XRP wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_live_bot_example | SOL | watch_recent_wallet_sell_flow | 194 | -4.69 | 4.69 | 2.14 | 2.54 | 0.000000 | 19.66 | label public_live_bot_example/SOL wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | HYPE | watch_recent_wallet_sell_flow | 33 | -128.54 | 128.54 | 5.38 | 123.16 | 0.000000 | 15.75 | label public_hypertracker_example/HYPE wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | xyz:GOLD | watch_recent_wallet_sell_flow | 10 | -140.76 | 140.76 | 4.98 | 135.78 | 0.000000 | 14.72 | label public_hypertracker_example/xyz:GOLD wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | APT | watch_wallet_long_pressure | 37 | 1196.41 | 6.95 | 3.65 | 3.30 | 534.630000 | 5.59 | label public_hypertracker_example/APT wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | TON | watch_recent_wallet_sell_flow | 6 | -45.75 | 45.74 | 0.84 | 44.90 | 0.000000 | 5.14 | label public_hypertracker_example/TON wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | xyz:NATGAS | watch_recent_wallet_sell_flow | 9 | -16.09 | 16.09 | 0.35 | 15.74 | 0.000000 | 2.49 | label public_hypertracker_example/xyz:NATGAS wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | xyz:SILVER | watch_recent_wallet_sell_flow | 4 | -17.03 | 17.03 | 0.17 | 16.86 | 0.000000 | 2.10 | label public_hypertracker_example/xyz:SILVER wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | xyz:MSTR | watch_recent_wallet_buy_flow | 8 | 993.83 | 6.17 | 3.75 | 2.42 | 0.000000 | 2.04 | label public_hypertracker_example/xyz:MSTR wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | SEI | watch_recent_wallet_sell_flow | 5 | -11.04 | 11.04 | 0.38 | 10.66 | 0.000000 | 1.58 | label public_hypertracker_example/SEI wallet-flow pressure over 15m/1h/4h, then compare against market-wide flow and execution costs |
| public_hypertracker_example | @150 | reject_negative_wallet_pnl | 9 | -673.28 | -1.58 | 1.10 | -2.68 | 0.000000 | -2.68 | do not follow public_hypertracker_example on @150; keep only as negative-control wallet-flow sample |
| public_hypertracker_example | SOL | reject_negative_wallet_pnl | 2 | 1.37 | -1.37 | 1.76 | -3.12 | 0.000000 | -3.12 | do not follow public_hypertracker_example on SOL; keep only as negative-control wallet-flow sample |
| public_live_bot_example | ETH | reject_negative_wallet_pnl | 207 | 28.92 | -3.63 | 2.68 | -6.31 | 0.015000 | -6.31 | do not follow public_live_bot_example on ETH; keep only as negative-control wallet-flow sample |
| public_live_bot_example | BTC | reject_negative_wallet_pnl | 241 | 77.22 | -25.88 | 3.11 | -28.99 | 0.000850 | -28.99 | do not follow public_live_bot_example on BTC; keep only as negative-control wallet-flow sample |
| public_hypertracker_example | BTC | reject_negative_wallet_pnl | 45 | 7922.82 | -250.17 | 15.47 | -265.64 | 0.120000 | -265.64 | do not follow public_hypertracker_example on BTC; keep only as negative-control wallet-flow sample |
| public_hypertracker_example | ASTER | reject_negative_wallet_pnl | 27 | 759.06 | -759.06 | 2.02 | -761.08 | 0.000000 | -761.08 | do not follow public_hypertracker_example on ASTER; keep only as negative-control wallet-flow sample |

## Caveat

These rows are useful only as seed observations. A wallet-flow alpha still needs entity selection, survivorship checks, forward labels, costs, and anti-copycat risk controls.
