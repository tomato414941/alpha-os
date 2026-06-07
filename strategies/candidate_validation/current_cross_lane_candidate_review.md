# Current Cross-Lane Candidate Review

This consolidates current candidate screens and first short-horizon labels. It is a triage board, not a deployable strategy ranking.

| asset | score | lanes | positive labels | negative labels | pending labels | note |
| --- | ---: | --- | --- | --- | --- | --- |
| BEAT | 4.0603 | okx_pressure; okx_liquidation | liq_cont15=0.0240 |  | okx_pressure15 | first labels support follow-up |
| MEGA | 3.7050 | hl_candidate_label; okx_pressure; on_chain_flow | hl15=0.0178 |  | okx_pressure15; chain15; chain15 | first labels support follow-up |
| BTC | 3.1495 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | liq_cont15=0.0003; l2_imbalance15=0.0001 |  | okx_pressure15; chain15; chain15 | first labels support follow-up |
| WLD | 2.9989 | hl_candidate_label; okx_pressure; okx_liquidation | hl15=0.0197 |  | okx_pressure15; liq_cont15 | first labels support follow-up |
| IP | 2.8057 | hl_candidate_label; okx_pressure | hl15=0.0160 |  | okx_pressure15 | first labels support follow-up |
| JTO | 2.7076 | okx_pressure; okx_liquidation; l2_imbalance_monitor | liq_cont15=0.0023; l2_imbalance15=0.0125 |  | okx_pressure15 | first labels support follow-up |
| XMR | 2.5530 | hl_candidate_label | hl15=0.0111 |  |  | first labels support follow-up |
| ZORA | 2.2743 | hl_candidate_label | hl15=0.0055 |  |  | first labels support follow-up |
| KAITO | 2.1882 | hl_candidate_label | hl15=0.0038 |  |  | first labels support follow-up |
| AIXBT | 2.1603 | hl_candidate_label | hl15=0.0032 |  |  | first labels support follow-up |
| APEX | 2.1524 | hl_candidate_label | hl15=0.0030 |  |  | first labels support follow-up |
| BSV | 2.0874 | hl_candidate_label | hl15=0.0017 |  |  | first labels support follow-up |
| ZRO | 2.0633 | hl_candidate_label | hl15=0.0013 |  |  | first labels support follow-up |
| SAGA | 2.0365 | hl_candidate_label | hl15=0.0007 |  |  | first labels support follow-up |
| HOME | 2.0000 | okx_pressure; okx_liquidation |  |  | okx_pressure15; liq_cont15 | waiting for elapsed labels |
| XLM | 1.9474 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0122 | liq_cont15=-0.0039 | okx_pressure15; chain15; chain15 | mixed evidence; isolate which source is real |
| XPL | 1.6405 | okx_pressure; l2_imbalance_monitor; sector_perp_context | l2_imbalance15=0.0030 |  | okx_pressure15; sector_perp15=:Echo Launchpad | first labels support follow-up |
| OPN | 1.5173 | okx_pressure; okx_liquidation | liq_cont15=0.0047 |  | okx_pressure15 | first labels support follow-up |
| ADA | 1.3750 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0015 | liq_cont15=-0.0006 | okx_pressure15; chain15; chain15 | mixed evidence; isolate which source is real |
| H | 1.2848 | okx_pressure; okx_liquidation | liq_cont15=0.0026 |  | okx_pressure15 | first labels support follow-up |
| ZEC | 1.2427 | okx_pressure; okx_liquidation; sector_perp_context |  |  | okx_pressure15; liq_cont15; sector_perp15=:Privacy; sector_perp15=:Zero Knowledge (ZK) | waiting for elapsed labels |
| SOL | 1.0779 | okx_pressure; l2_imbalance_monitor; on_chain_flow |  | l2_imbalance15=-0.0033 | okx_pressure15; chain15; chain15 | current short labels are weak |
| SUI | 0.8972 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow |  | liq_cont15=-0.0004; l2_imbalance15=-0.0019 | okx_pressure15; chain15; chain15 | current short labels are weak |
| AVAX | 0.8341 | okx_pressure; on_chain_flow |  |  | okx_pressure15; chain15; chain15 | waiting for elapsed labels |
| ETH | 0.7915 | okx_pressure; okx_liquidation; l2_imbalance_monitor; on_chain_flow | l2_imbalance15=0.0010 | liq_cont15=-0.0014 | okx_pressure15; chain15; chain15 | mixed evidence; isolate which source is real |

## Interpretation

Higher score means more current evidence survived a short label or appeared in multiple lanes. A negative label does not kill a candidate if another PnL component, such as funding, is still unmodeled.
