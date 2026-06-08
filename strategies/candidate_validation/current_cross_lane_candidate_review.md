# Current Cross-Lane Candidate Review

This consolidates current candidate screens and first short-horizon labels. It is a triage board, not a deployable strategy ranking.

| asset | score | lanes | positive labels | negative labels | pending labels | note |
| --- | ---: | --- | --- | --- | --- | --- |
| NEAR | 5.3498 | okx_pressure; exchange_catalyst; protocol_activity; on_chain_flow | exchange15=0.0250:network_event_watch; chain15=0.0090:HL:chain_flow_reversal_watch; chain15=0.0083:OKX:chain_flow_reversal_watch |  | okx_pressure15; protocol15=:protocol_activity_watch | first labels support follow-up |
| MEGA | 3.4940 | okx_pressure; exchange_catalyst; on_chain_flow | exchange15=0.0694:spot_listing_watch; chain15=0.0031:OKX:chain_outflow_stress_watch; chain15=0.0023:HL:chain_outflow_stress_watch | exchange15=-0.0281:spot_listing_watch | okx_pressure15 | mixed evidence; isolate which source is real |
| CHIP | 2.9957 | exchange_catalyst | exchange15=0.0065:spot_listing_watch |  | exchange15=0.0000:perp_listing_watch; exchange15=0.0000:spot_listing_watch; exchange15=0.0000:spot_listing_watch | first labels support follow-up |
| ADA | 2.9842 | okx_pressure; l2_imbalance_monitor; on_chain_flow | chain15=0.0078:HL:chain_flow_reversal_watch; chain15=0.0073:OKX:chain_flow_reversal_watch | l2_imbalance15=-0.0078 | okx_pressure15 | mixed evidence; isolate which source is real |
| BTC | 2.5455 | okx_pressure; okx_liquidation; protocol_activity; on_chain_flow | chain15=0.0043:HL:chain_flow_reversal_watch; chain15=0.0041:OKX:chain_flow_reversal_watch | liq_cont15=-0.0022 | okx_pressure15; protocol15=:protocol_activity_watch | mixed evidence; isolate which source is real |
| SOL | 2.2019 | okx_pressure; exchange_catalyst; on_chain_flow | exchange15=0.0013:exchange_removal_watch; chain15=0.0055:HL:chain_flow_reversal_watch; chain15=0.0051:OKX:chain_flow_reversal_watch |  | okx_pressure15 | first labels support follow-up |
| HYPE | 2.1935 | okx_pressure; okx_liquidation; on_chain_flow | chain15=0.0243:HL:chain_flow_reversal_watch; chain15=0.0243:OKX:chain_flow_reversal_watch | liq_cont15=-0.0046 | okx_pressure15 | mixed evidence; isolate which source is real |
| HOME | 2.0000 | okx_pressure; okx_liquidation |  |  | okx_pressure15; liq_cont15 | waiting for elapsed labels |
| ETH | 1.9347 | okx_pressure; okx_liquidation; on_chain_flow | chain15=0.0072:HL:chain_flow_reversal_watch; chain15=0.0069:OKX:chain_flow_reversal_watch |  | okx_pressure15; liq_cont15 | first labels support follow-up |
| SEI | 1.8994 | hl_candidate_label; on_chain_flow | chain15=0.0077:OKX:chain_flow_reversal_watch; chain15=0.0070:HL:chain_flow_reversal_watch |  | hl15 | first labels support follow-up |
| MON | 1.8131 | okx_pressure; on_chain_flow | chain15=0.0072:OKX:chain_flow_reversal_watch; chain15=0.0064:HL:chain_flow_reversal_watch |  | okx_pressure15 | first labels support follow-up |
| BNB | 1.8127 | okx_pressure; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0036; chain15=0.0036:HL:chain_flow_reversal_watch; chain15=0.0033:OKX:chain_flow_reversal_watch |  | okx_pressure15 | first labels support follow-up |
| POL | 1.7572 | sector_perp_context; exchange_catalyst; on_chain_flow | sector_perp15=0.0035:Zero Knowledge (ZK); chain15=0.0035:HL:chain_flow_reversal_watch; chain15=0.0034:OKX:chain_flow_reversal_watch | exchange15=-0.0008:network_event_watch | exchange15=0.0000:network_event_watch | mixed evidence; isolate which source is real |
| ARB | 1.7070 | okx_pressure; on_chain_flow | chain15=0.0058:HL:chain_flow_reversal_watch; chain15=0.0053:OKX:chain_flow_reversal_watch |  | okx_pressure15 | first labels support follow-up |
| AI | 1.7045 | okx_pressure; exchange_catalyst |  |  | okx_pressure15; exchange15=:spot_listing_watch; exchange15=:spot_listing_watch; exchange15=:spot_listing_watch; exchange15=:exchange_removal_watch | waiting for elapsed labels |
| SUI | 1.6307 | okx_pressure; okx_liquidation; on_chain_flow | chain15=0.0049:OKX:chain_flow_reversal_watch; chain15=0.0049:HL:chain_flow_reversal_watch |  | okx_pressure15; liq_cont15 | first labels support follow-up |
| ZEC | 1.5610 | okx_pressure; okx_liquidation; sector_perp_context; protocol_activity |  | liq_cont15=-0.0090; sector_perp15=-0.0019:Privacy; sector_perp15=-0.0019:Zero Knowledge (ZK); protocol15=-0.0071:protocol_activity_funding_overlap | okx_pressure15 | current short labels are weak |
| STX | 1.4022 | on_chain_flow | chain15=0.0040:HL:chain_flow_reversal_watch; chain15=0.0038:OKX:chain_flow_reversal_watch |  |  | first labels support follow-up |
| APT | 1.3984 | okx_pressure; on_chain_flow | chain15=0.0046:HL:chain_flow_reversal_watch; chain15=0.0045:OKX:chain_flow_reversal_watch |  | okx_pressure15 | first labels support follow-up |
| XMR | 1.3874 | hl_candidate_label; l2_imbalance_monitor | l2_imbalance15=0.0032 |  | hl15 | first labels support follow-up |
| WLD | 1.3679 | hl_candidate_label; okx_pressure; okx_liquidation; l2_imbalance_monitor | l2_imbalance15=0.0015 |  | hl15; okx_pressure15; liq_cont15 | first labels support follow-up |
| STRK | 1.3542 | on_chain_flow | chain15=0.0059:HL:chain_flow_reversal_watch; chain15=0.0054:OKX:chain_flow_reversal_watch |  |  | first labels support follow-up |
| BERA | 1.2612 | on_chain_flow | chain15=0.0031:HL:chain_flow_reversal_watch; chain15=0.0029:OKX:chain_flow_reversal_watch |  |  | first labels support follow-up |
| PEPE | 1.1830 | okx_pressure; okx_liquidation | liq_cont15=0.0018 |  | okx_pressure15 | first labels support follow-up |
| LINK | 1.0784 | okx_pressure; protocol_activity |  | protocol15=-0.0010:protocol_activity_watch | okx_pressure15 | current short labels are weak |

## Interpretation

Higher score means more current evidence survived a short label or appeared in multiple lanes. A negative label does not kill a candidate if another PnL component, such as funding, is still unmodeled.
