# Current Symbol Cluster Conflicts

This separates symbol clusters that confirm one direction from clusters that mix directional, relative-value, yield, and risk-avoidance ideas. It is a conflict screen, not a trade list.

| symbol | status | score | sources | candidates | bias counts | dominant | top opportunities | next step |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| BTC | mixed_structure_conflict | 135.6078 | 9 | 11 | L=1, S=0, RV=9, Y=0, R=1, N=0 | relative_value | btc_short_put_spread_20260609, btc_short_put_spread_20260610, mstr_btc_relative_value, btc_risk_off_short_stack, btc-26mar27_basis | split BTC labels by lane before trading; do not collapse conflicting ideas into one action |
| ETH | mixed_structure_conflict | 125.7968 | 7 | 10 | L=2, S=0, RV=5, Y=0, R=0, N=3 | relative_value | eth-12jun26_basis, eth-26mar27_basis, eth_attention_price_context, eth_calendar_spread_20260612, eth_institutional_flow_news_event | split ETH labels by lane before trading; do not collapse conflicting ideas into one action |
| SOL | mixed_direction_conflict | 123.6172 | 5 | 7 | L=3, S=2, RV=0, Y=0, R=2, N=0 | long | solana_stablecoin_migration, coinup.io_(futures)_sol-usdt_positioning, kucoin_futures_solusdtm_positioning, sol_attention_price_context, solana_bountywork_sol_dex_pool | split SOL labels by lane before trading; do not collapse conflicting ideas into one action |
| ZEC | mixed_direction_conflict | 120.8110 | 4 | 6 | L=2, S=1, RV=2, Y=0, R=0, N=1 | mixed | whitebit_futures_zec_perp_positioning, zec_attention_price_context, zec_institutional_flow_news_event, zec_security_risk_news_event, zec_attention_funding_overlap | split ZEC labels by lane before trading; do not collapse conflicting ideas into one action |
| JUP | confirmed_long_cluster | 118.9345 | 2 | 3 | L=3, S=0, RV=0, Y=0, R=0, N=0 | long | jup_fee_growth_price_context, jup_fee_yield_valuation, jup_protocol_fee_growth | paper-label JUP long setup against forward return, costs, depth, and failure regime |
| AAVE | confirmed_long_cluster | 118.2895 | 2 | 3 | L=3, S=0, RV=0, Y=0, R=0, N=0 | long | aave_fee_growth_price_context, aave_fee_yield_valuation, aave_protocol_fee_growth | paper-label AAVE long setup against forward return, costs, depth, and failure regime |
| ONDO | yield_cluster | 117.5000 | 3 | 6 | L=0, S=0, RV=1, Y=4, R=1, N=0 | yield | usdy_stablecoin_peg_stress, ethereum_ondo-yield-assets_usdy_yield_peg, stellar_ondo-yield-assets_usdy_yield_peg, sei_ondo-yield-assets_usdy_yield_peg, solana_ondo-yield-assets_usdy_yield_peg | validate ONDO yield mechanics, venue access, liquidity, fees, and unwind path |
| UNI | confirmed_long_cluster | 117.2911 | 2 | 3 | L=3, S=0, RV=0, Y=0, R=0, N=0 | long | uni_fee_growth_price_context, uni_fee_yield_valuation, uni_protocol_fee_growth | paper-label UNI long setup against forward return, costs, depth, and failure regime |
| HYPE | mixed_direction_conflict | 116.2663 | 3 | 3 | L=1, S=1, RV=0, Y=0, R=0, N=1 | mixed | hype_unlock_event, hype_protocol_fee_growth, hyperliquid_l1_stablecoin_migration | split HYPE labels by lane before trading; do not collapse conflicting ideas into one action |
| MORPHO | confirmed_long_cluster | 115.5860 | 2 | 3 | L=3, S=0, RV=0, Y=0, R=0, N=0 | long | morpho_fee_growth_price_context, morpho_fee_yield_valuation, morpho_protocol_fee_growth | paper-label MORPHO long setup against forward return, costs, depth, and failure regime |
| CRV | confirmed_long_cluster | 112.9175 | 2 | 2 | L=2, S=0, RV=0, Y=0, R=0, N=0 | long | crv_fee_growth_price_context, crv_fee_yield_valuation | paper-label CRV long setup against forward return, costs, depth, and failure regime |
| USDY | yield_cluster | 109.0000 | 2 | 5 | L=0, S=0, RV=0, Y=4, R=1, N=0 | yield | usdy_stablecoin_peg_stress, ethereum_ondo-yield-assets_usdy_yield_peg, stellar_ondo-yield-assets_usdy_yield_peg, sei_ondo-yield-assets_usdy_yield_peg, solana_ondo-yield-assets_usdy_yield_peg | validate USDY yield mechanics, venue access, liquidity, fees, and unwind path |
| ZRO | mixed_structure_conflict | 107.6561 | 2 | 2 | L=0, S=1, RV=1, Y=0, R=0, N=0 | mixed | zro_unlock_event, zro_perp_crowding_reversion | split ZRO labels by lane before trading; do not collapse conflicting ideas into one action |
| PENDLE | multi_source_watch | 103.4239 | 2 | 1 | L=1, S=0, RV=0, Y=0, R=0, N=0 | long | pendle_fee_growth_price_context | collect more PENDLE snapshots before treating this as a cluster |
| REUSD | multi_source_watch | 101.5000 | 2 | 2 | L=0, S=0, RV=0, Y=1, R=1, N=0 | mixed | reusd_stablecoin_peg_stress, ethereum_re_reusd_yield_peg | collect more REUSD snapshots before treating this as a cluster |
| APXUSD | multi_source_watch | 101.5000 | 2 | 2 | L=0, S=0, RV=0, Y=1, R=1, N=0 | mixed | apxusd_stablecoin_peg_stress, ethereum_apyx-protocol_apxusd_yield_peg | collect more APXUSD snapshots before treating this as a cluster |
| MSTR | multi_source_watch | 96.1790 | 2 | 1 | L=0, S=0, RV=1, Y=0, R=0, N=0 | relative_value | mstr_btc_relative_value | collect more MSTR snapshots before treating this as a cluster |
| JTO | multi_source_watch | 95.5063 | 2 | 2 | L=0, S=0, RV=1, Y=0, R=0, N=1 | mixed | jto_l2_imbalance_probe, jto_okx_liquidation_continuation | collect more JTO snapshots before treating this as a cluster |
| PAXG | multi_source_watch | 94.6916 | 2 | 2 | L=0, S=0, RV=1, Y=1, R=0, N=0 | mixed | ethereum_usdc_paxg_lending_pressure, paxg_perp_crowding_reversion | collect more PAXG snapshots before treating this as a cluster |
| PMUSD | single_candidate_watch | 93.0000 | 1 | 1 | L=0, S=0, RV=0, Y=0, R=1, N=0 | risk_or_avoid | pmusd_stablecoin_peg_stress | collect more PMUSD snapshots before treating this as a cluster |
| USYC | single_candidate_watch | 93.0000 | 1 | 1 | L=0, S=0, RV=0, Y=0, R=1, N=0 | risk_or_avoid | usyc_stablecoin_peg_stress | collect more USYC snapshots before treating this as a cluster |
| TRX | multi_source_watch | 89.7201 | 2 | 2 | L=0, S=1, RV=0, Y=0, R=0, N=1 | mixed | bybit_(futures)_trxusdt_positioning, tron_stablecoin_migration | collect more TRX snapshots before treating this as a cluster |
| PENGU | multi_source_watch | 87.6745 | 2 | 1 | L=1, S=0, RV=0, Y=0, R=0, N=0 | long | pengu_attention_price_context | collect more PENGU snapshots before treating this as a cluster |
| TAO | multi_source_watch | 86.7836 | 2 | 1 | L=1, S=0, RV=0, Y=0, R=0, N=0 | long | tao_attention_price_context | collect more TAO snapshots before treating this as a cluster |
| NEAR | multi_source_watch | 86.6876 | 2 | 1 | L=1, S=0, RV=0, Y=0, R=0, N=0 | long | near_attention_price_context | collect more NEAR snapshots before treating this as a cluster |
| BONDUSD | single_candidate_watch | 86.3000 | 1 | 1 | L=0, S=0, RV=0, Y=1, R=0, N=0 | yield | ethereum_usr_bondusd_lending_pressure | collect more BONDUSD snapshots before treating this as a cluster |
| USR | single_candidate_watch | 86.3000 | 1 | 1 | L=0, S=0, RV=0, Y=1, R=0, N=0 | yield | ethereum_usr_bondusd_lending_pressure | collect more USR snapshots before treating this as a cluster |
| DOLA | single_candidate_watch | 86.1846 | 1 | 1 | L=0, S=0, RV=0, Y=0, R=1, N=0 | risk_or_avoid | dola_stablecoin_peg_stress | collect more DOLA snapshots before treating this as a cluster |
| SDEUSD | single_candidate_watch | 85.7425 | 1 | 1 | L=0, S=0, RV=0, Y=1, R=0, N=0 | yield | ethereum_usdc_sdeusd_lending_pressure | collect more SDEUSD snapshots before treating this as a cluster |
| WSTUSR | single_candidate_watch | 85.1216 | 1 | 1 | L=0, S=0, RV=0, Y=1, R=0, N=0 | yield | ethereum_usdc_wstusr_lending_pressure | collect more WSTUSR snapshots before treating this as a cluster |
| HERMES | single_candidate_watch | 85.1211 | 1 | 1 | L=0, S=0, RV=0, Y=1, R=0, N=0 | yield | base_usdc_hermes_lending_pressure | collect more HERMES snapshots before treating this as a cluster |
| STABLE | multi_source_watch | 84.0140 | 1 | 2 | L=0, S=0, RV=0, Y=0, R=0, N=2 | neutral | stable_bybitperp_hlperp_funding, stable_okxswap_hlperp_funding | collect more STABLE snapshots before treating this as a cluster |
| SAGA | multi_source_watch | 83.0354 | 1 | 2 | L=0, S=0, RV=0, Y=0, R=0, N=2 | neutral | saga_hlperp_bybitperp_funding, saga_hlperp_cash_or_spot_proxy_funding | collect more SAGA snapshots before treating this as a cluster |
| SKHX | single_candidate_watch | 81.3085 | 1 | 1 | L=0, S=1, RV=0, Y=0, R=0, N=0 | short | hyperliquid_(futures)_xyz:skhx-usd_positioning | collect more SKHX snapshots before treating this as a cluster |
| ATOM | single_candidate_watch | 81.2728 | 1 | 1 | L=0, S=1, RV=0, Y=0, R=0, N=0 | short | lbank_(futures)_atomusdt_positioning | collect more ATOM snapshots before treating this as a cluster |
| KNEIRO | single_candidate_watch | 80.2323 | 1 | 1 | L=0, S=0, RV=0, Y=0, R=0, N=1 | neutral | kneiro_hlperp_bybitperp_funding | collect more KNEIRO snapshots before treating this as a cluster |
| MON | multi_source_watch | 79.4245 | 1 | 2 | L=0, S=0, RV=0, Y=0, R=0, N=2 | neutral | okx_mon_chain_flow_context, mon_chain_flow_context | collect more MON snapshots before treating this as a cluster |
| BOUNTYWORK | single_candidate_watch | 78.8561 | 1 | 1 | L=0, S=0, RV=0, Y=0, R=1, N=0 | risk_or_avoid | solana_bountywork_sol_dex_pool | collect more BOUNTYWORK snapshots before treating this as a cluster |
| BOUTYWORK | single_candidate_watch | 78.5212 | 1 | 1 | L=0, S=0, RV=0, Y=0, R=1, N=0 | risk_or_avoid | solana_boutywork_sol_dex_pool | collect more BOUTYWORK snapshots before treating this as a cluster |
| PRIORITY | multi_source_watch | 77.9361 | 1 | 2 | L=0, S=0, RV=0, Y=0, R=0, N=2 | neutral | okx_eth_chain_flow_context, eth_chain_flow_context | collect more PRIORITY snapshots before treating this as a cluster |

## Interpretation

`confirmed_*` rows are the cleanest next paper-label candidates. `mixed_*` rows are often more interesting, but they need a label that separates which lane is actually driving returns before any action.
