# Current Seed Wallet Flow Actionability

This filters public seed-wallet flow into paper-label candidates. It is not a copy-trading rule and not a verified entity model.

| candidate | asset | side | status | score | fills | net buy USD | net PnL | position USD | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| public_hypertracker_example_eth_wallet_flow_actionability | ETH | paper_long_wallet_flow | wallet_position_follow_candidate | 98.94 | 79 | 6903.69 | 414.58 | 4676.47 | seed wallet has positive realized PnL and an open tradable position |
| public_hypertracker_example_zec_wallet_flow_actionability | ZEC | paper_long_wallet_flow | wallet_position_follow_candidate | 83.37 | 40 | 49.59 | 443.76 | 563.93 | seed wallet has positive realized PnL and an open tradable position |
| public_hypertracker_example_apt_wallet_flow_actionability | APT | paper_long_wallet_flow | wallet_position_follow_candidate | 61.93 | 37 | 1196.41 | 3.30 | 358.15 | seed wallet has positive realized PnL and an open tradable position |
| public_hypertracker_example_hyna_eth_wallet_flow_actionability | ETH | paper_short_wallet_flow | wallet_flow_watch | 60.85 | 40 | -385.04 | 369.33 | 0.00 | seed wallet has positive realized PnL but the tradable flow is still small |
| public_hypertracker_example_hyna_btc_wallet_flow_actionability | BTC | paper_short_wallet_flow | wallet_flow_watch | 55.76 | 40 | -281.87 | 269.56 | 0.00 | seed wallet has positive realized PnL but the tradable flow is still small |
| public_live_bot_example_xrp_wallet_flow_actionability | XRP | paper_long_wallet_flow | wallet_flow_watch | 50.92 | 201 | 8.13 | 17.13 | 27.53 | seed wallet has positive realized PnL but the tradable flow is still small |
| public_live_bot_example_sol_wallet_flow_actionability | SOL | paper_short_wallet_flow | wallet_flow_watch | 49.83 | 194 | -4.69 | 2.54 | 0.00 | seed wallet has positive realized PnL but the tradable flow is still small |
| public_hypertracker_example_hype_wallet_flow_actionability | HYPE | paper_short_wallet_flow | wallet_flow_watch | 47.94 | 33 | -128.54 | 123.16 | 0.00 | seed wallet has positive realized PnL but the tradable flow is still small |
| public_hypertracker_example_xyz_cl_wallet_flow_actionability |  | paper_short_wallet_flow | wallet_flow_blocked_untradable_asset | 47.71 | 324 | -708.12 | 688.76 | 0.00 | source coin does not map to a current Hyperliquid tradable asset |
| public_hypertracker_example_ton_wallet_flow_actionability | TON | paper_short_wallet_flow | wallet_flow_watch | 42.59 | 6 | -45.75 | 44.90 | 0.00 | seed wallet has positive realized PnL but the tradable flow is still small |
| public_hypertracker_example_sei_wallet_flow_actionability | SEI | paper_short_wallet_flow | wallet_flow_watch | 40.79 | 5 | -11.04 | 10.66 | 0.00 | seed wallet has positive realized PnL but the tradable flow is still small |
| public_hypertracker_example_btc_wallet_flow_actionability | BTC | paper_long_wallet_flow | wallet_flow_reject_negative_seed_pnl | 22.17 | 45 | 7922.82 | -265.64 | 7586.52 | seed wallet row has negative realized PnL after fees |
| public_hypertracker_example_xyz_gold_wallet_flow_actionability |  | paper_short_wallet_flow | wallet_flow_blocked_untradable_asset | 19.43 | 10 | -140.76 | 135.78 | 0.00 | source coin does not map to a current Hyperliquid tradable asset |
| public_hypertracker_example_xyz_mstr_wallet_flow_actionability |  | paper_long_wallet_flow | wallet_flow_blocked_untradable_asset | 13.52 | 8 | 993.83 | 2.42 | 0.00 | source coin does not map to a current Hyperliquid tradable asset |
| public_hypertracker_example_xyz_natgas_wallet_flow_actionability |  | paper_short_wallet_flow | wallet_flow_blocked_untradable_asset | 13.25 | 9 | -16.09 | 15.74 | 0.00 | source coin does not map to a current Hyperliquid tradable asset |
| public_hypertracker_example_150_wallet_flow_actionability |  | paper_short_wallet_flow | wallet_flow_blocked_untradable_asset | 13.12 | 9 | -673.28 | -2.68 | 0.00 | source coin does not map to a current Hyperliquid tradable asset |
| public_hypertracker_example_xyz_silver_wallet_flow_actionability |  | paper_short_wallet_flow | wallet_flow_blocked_untradable_asset | 13.06 | 4 | -17.03 | 16.86 | 0.00 | source coin does not map to a current Hyperliquid tradable asset |
| public_live_bot_example_btc_wallet_flow_actionability | BTC | paper_long_wallet_flow | wallet_flow_reject_negative_seed_pnl | 10.18 | 241 | 77.22 | -28.99 | 53.74 | seed wallet row has negative realized PnL after fees |
| public_live_bot_example_eth_wallet_flow_actionability | ETH | paper_long_wallet_flow | wallet_flow_reject_negative_seed_pnl | 10.08 | 207 | 28.92 | -6.31 | 25.08 | seed wallet row has negative realized PnL after fees |
| public_hypertracker_example_aster_wallet_flow_actionability | ASTER | observe_wallet_flow | wallet_flow_reject_negative_seed_pnl | 2.11 | 27 | 759.06 | -761.08 | 0.00 | seed wallet row has negative realized PnL after fees |
| public_hypertracker_example_sol_wallet_flow_actionability | SOL | observe_wallet_flow | wallet_flow_reject_negative_seed_pnl | 0.10 | 2 | 1.37 | -3.12 | 0.00 | seed wallet row has negative realized PnL after fees |

## Rule

Only tradable assets with positive realized seed-wallet PnL become actionability candidates. Position-follow candidates need an open position; recent-flow candidates need enough fill history and a material net buy/sell imbalance.
