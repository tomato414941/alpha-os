# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.775983 | 320165522 | 0.2152 | 72940 | 0.013710 | tradable_context_ok | public venue context does not obviously block a small repeat |
| INJ | broad_alpha_paper | 93.1248 | -0.397063 | 4783377 | 2.2501 | 6652 | 0.150325 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 3076523 | 3.7695 | 6878 | 0.145401 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1877518 | 2.8404 | 9735 | 0.102718 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 83204052 | 5.1586 | 30726 | 0.032546 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.017083 | 253668184 | 0.1493 | 546226 | 0.001831 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.064097 | 6112818 | 1.6527 | 82038 | 0.012189 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.119376 | 1145710834 | 0.5910 | 12655836 | 0.000079 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2326264 | 5.1608 | 4938 | 0.202511 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | 0.009814 | 2730241 | 1.2143 | 5133 | 0.194833 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 36040795 | 0.1318 | 82640 | 0.012101 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.310190 | 10748857 | 2.3359 | 83214 | 0.012017 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | 0.109500 | 2449692 | 4.4402 | 11488 | 0.087047 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.042534 | 2908760222 | 0.1580 | 2324241 | 0.000430 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1055764776 | 1.4225 | 125507 | 0.007968 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7266430 | 0.3138 | 22804 | 0.043852 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.097039 | 86096619 | 2.0059 | 21553 | 0.046397 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 867988 | 5.5985 | 5446 | 0.183638 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.095748 | 801004 | 2.2183 | 9240 | 0.108229 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 172897 | 6.4788 | 2608 | 0.383494 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | -0.209536 | 951295 | 5.8207 | 11211 | 0.089200 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 830817 | 5.7671 | 2044 | 0.489229 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.109500 | 785050 | 7.2498 | 394 | 2.539487 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.045601 | 588761 | 4.7073 | 5614 | 0.178137 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | -0.051859 | 235423 | 2.3703 | 7424 | 0.134698 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1910820 | 7.7141 | 1116 | 0.896111 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
