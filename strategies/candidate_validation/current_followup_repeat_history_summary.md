# Current Follow-Up Repeat History Summary

This aggregates stored repeat labels without blending source meaning. Rows with pending observations should be rerun after the 15m horizon matures.

- total groups: `95`
- repeat-priority groups: `6`

| group type | group | labeled | pending | hit 15m | mean dir15 | min dir15 | max dir15 | action | evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| asset_source | XLM/okx_pressure | 2 | 2 | 1.000 | 0.004693 | 0.004377 | 0.005009 | repeat_priority | examples=HL/XLM/okx_pressure;OKX/XLM/okx_pressure; pending=2 |
| asset_source | XLM/l2_imbalance | 2 | 2 | 1.000 | 0.004693 | 0.004377 | 0.005009 | repeat_priority | examples=HL/XLM/l2_imbalance;OKX/XLM/l2_imbalance; pending=2 |
| asset_source | PUMP/liquidation | 2 | 2 | 1.000 | 0.001993 | 0.001992 | 0.001993 | repeat_priority | examples=OKX/PUMP/liquidation;HL/PUMP/liquidation; pending=2 |
| asset_source | PUMP/sector_rotation | 2 | 2 | 1.000 | 0.001993 | 0.001992 | 0.001993 | repeat_priority | examples=OKX/PUMP/sector_rotation;HL/PUMP/sector_rotation; pending=2 |
| asset_source | XRP/okx_pressure | 2 | 2 | 1.000 | 0.001488 | 0.001225 | 0.001751 | repeat_priority | examples=OKX/XRP/okx_pressure;HL/XRP/okx_pressure; pending=2 |
| asset_source | XRP/liquidation | 2 | 2 | 1.000 | 0.001488 | 0.001225 | 0.001751 | repeat_priority | examples=OKX/XRP/liquidation;HL/XRP/liquidation; pending=2 |
| venue_source | OKX/liquidation | 12 | 12 | 0.667 | 0.002951 | -0.002898 | 0.017745 | mixed_continue_sampling | examples=OKX/JTO/liquidation;OKX/H/liquidation;OKX/ALLO/liquidation; pending=12 |
| source | liquidation | 21 | 21 | 0.714 | 0.001868 | -0.002898 | 0.017745 | mixed_continue_sampling | examples=OKX/JTO/liquidation;OKX/H/liquidation;OKX/ALLO/liquidation; pending=21 |
| venue_source | HL/l2_imbalance | 4 | 4 | 0.750 | 0.000915 | -0.001905 | 0.005009 | mixed_continue_sampling | examples=HL/XLM/l2_imbalance;HL/BTC/l2_imbalance;HL/ETH/l2_imbalance; pending=4 |
| venue_source | HL/okx_pressure | 7 | 7 | 0.571 | 0.000713 | -0.000736 | 0.005009 | mixed_continue_sampling | examples=HL/XLM/okx_pressure;HL/XRP/okx_pressure;HL/WLD/okx_pressure; pending=7 |
| source | okx_pressure | 15 | 15 | 0.467 | 0.000533 | -0.001161 | 0.005009 | mixed_continue_sampling | examples=HL/XLM/okx_pressure;OKX/XLM/okx_pressure;OKX/XRP/okx_pressure; pending=15 |
| venue_source | HL/liquidation | 9 | 9 | 0.778 | 0.000424 | -0.001449 | 0.001992 | mixed_continue_sampling | examples=HL/PUMP/liquidation;HL/XRP/liquidation;HL/LTC/liquidation; pending=9 |
| venue_source | OKX/okx_pressure | 8 | 8 | 0.375 | 0.000377 | -0.001161 | 0.004377 | mixed_continue_sampling | examples=OKX/XLM/okx_pressure;OKX/XRP/okx_pressure;OKX/ETH/okx_pressure; pending=8 |
| asset_source | WLD/okx_pressure | 2 | 2 | 0.500 | -0.000145 | -0.000414 | 0.000124 | mixed_continue_sampling | examples=HL/WLD/okx_pressure;OKX/WLD/okx_pressure; pending=2 |
| asset_source | WLD/liquidation | 2 | 2 | 0.500 | -0.000145 | -0.000414 | 0.000124 | mixed_continue_sampling | examples=HL/WLD/liquidation;OKX/WLD/liquidation; pending=2 |
| venue_source | HL/sector_rotation | 3 | 3 | 0.333 | -0.000454 | -0.001905 | 0.001992 | mixed_continue_sampling | examples=HL/PUMP/sector_rotation;HL/ONDO/sector_rotation;HL/XPL/sector_rotation; pending=3 |
| source | sector_rotation | 6 | 6 | 0.333 | -0.000572 | -0.002898 | 0.001993 | mixed_continue_sampling | examples=OKX/PUMP/sector_rotation;HL/PUMP/sector_rotation;OKX/XPL/sector_rotation; pending=6 |
| venue_source | OKX/sector_rotation | 3 | 3 | 0.333 | -0.000690 | -0.002898 | 0.001993 | mixed_continue_sampling | examples=OKX/PUMP/sector_rotation;OKX/XPL/sector_rotation;OKX/ONDO/sector_rotation; pending=3 |
| source | l2_imbalance | 8 | 8 | 0.625 | -0.001343 | -0.017745 | 0.005009 | mixed_continue_sampling | examples=HL/XLM/l2_imbalance;OKX/XLM/l2_imbalance;HL/BTC/l2_imbalance; pending=8 |
| venue_source | OKX/l2_imbalance | 4 | 4 | 0.500 | -0.003601 | -0.017745 | 0.004377 | mixed_continue_sampling | examples=OKX/XLM/l2_imbalance;OKX/ETH/l2_imbalance;OKX/XPL/l2_imbalance; pending=4 |
| asset_source | TON/liquidation | 2 | 2 | 1.000 | 0.000929 | 0.000696 | 0.001161 | keep_sampling | examples=OKX/TON/liquidation;HL/TON/liquidation; pending=2 |
| asset_source | LTC/liquidation | 2 | 2 | 1.000 | 0.000843 | 0.000736 | 0.000950 | keep_sampling | examples=OKX/LTC/liquidation;HL/LTC/liquidation; pending=2 |
| asset_source | ETH/okx_pressure | 2 | 2 | 1.000 | 0.000125 | 0.000122 | 0.000129 | keep_sampling | examples=OKX/ETH/okx_pressure;HL/ETH/okx_pressure; pending=2 |
| asset_source | ETH/liquidation | 2 | 2 | 1.000 | 0.000125 | 0.000122 | 0.000129 | keep_sampling | examples=OKX/ETH/liquidation;HL/ETH/liquidation; pending=2 |
| asset_source | ETH/l2_imbalance | 2 | 2 | 1.000 | 0.000125 | 0.000122 | 0.000129 | keep_sampling | examples=OKX/ETH/l2_imbalance;HL/ETH/l2_imbalance; pending=2 |
| asset_source | JTO/liquidation | 1 | 1 | 1.000 | 0.017745 | 0.017745 | 0.017745 | wait_for_second_label | examples=OKX/JTO/liquidation; pending=1 |
| venue_asset_source | OKX/JTO/liquidation | 1 | 1 | 1.000 | 0.017745 | 0.017745 | 0.017745 | wait_for_second_label | examples=OKX/JTO/liquidation; pending=1 |
| asset_source | H/liquidation | 1 | 1 | 1.000 | 0.013110 | 0.013110 | 0.013110 | wait_for_second_label | examples=OKX/H/liquidation; pending=1 |
| venue_asset_source | OKX/H/liquidation | 1 | 1 | 1.000 | 0.013110 | 0.013110 | 0.013110 | wait_for_second_label | examples=OKX/H/liquidation; pending=1 |
| venue_asset_source | HL/XLM/okx_pressure | 1 | 1 | 1.000 | 0.005009 | 0.005009 | 0.005009 | wait_for_second_label | examples=HL/XLM/okx_pressure; pending=1 |
| venue_asset_source | HL/XLM/l2_imbalance | 1 | 1 | 1.000 | 0.005009 | 0.005009 | 0.005009 | wait_for_second_label | examples=HL/XLM/l2_imbalance; pending=1 |
| venue_asset_source | OKX/XLM/okx_pressure | 1 | 1 | 1.000 | 0.004377 | 0.004377 | 0.004377 | wait_for_second_label | examples=OKX/XLM/okx_pressure; pending=1 |
| venue_asset_source | OKX/XLM/l2_imbalance | 1 | 1 | 1.000 | 0.004377 | 0.004377 | 0.004377 | wait_for_second_label | examples=OKX/XLM/l2_imbalance; pending=1 |
| asset_source | ALLO/liquidation | 1 | 1 | 1.000 | 0.002598 | 0.002598 | 0.002598 | wait_for_second_label | examples=OKX/ALLO/liquidation; pending=1 |
| venue_asset_source | OKX/ALLO/liquidation | 1 | 1 | 1.000 | 0.002598 | 0.002598 | 0.002598 | wait_for_second_label | examples=OKX/ALLO/liquidation; pending=1 |
| venue_asset_source | OKX/PUMP/liquidation | 1 | 1 | 1.000 | 0.001993 | 0.001993 | 0.001993 | wait_for_second_label | examples=OKX/PUMP/liquidation; pending=1 |
| venue_asset_source | OKX/PUMP/sector_rotation | 1 | 1 | 1.000 | 0.001993 | 0.001993 | 0.001993 | wait_for_second_label | examples=OKX/PUMP/sector_rotation; pending=1 |
| venue_asset_source | HL/PUMP/liquidation | 1 | 1 | 1.000 | 0.001992 | 0.001992 | 0.001992 | wait_for_second_label | examples=HL/PUMP/liquidation; pending=1 |
| venue_asset_source | HL/PUMP/sector_rotation | 1 | 1 | 1.000 | 0.001992 | 0.001992 | 0.001992 | wait_for_second_label | examples=HL/PUMP/sector_rotation; pending=1 |
| venue_asset_source | OKX/XRP/okx_pressure | 1 | 1 | 1.000 | 0.001751 | 0.001751 | 0.001751 | wait_for_second_label | examples=OKX/XRP/okx_pressure; pending=1 |
| venue_asset_source | OKX/XRP/liquidation | 1 | 1 | 1.000 | 0.001751 | 0.001751 | 0.001751 | wait_for_second_label | examples=OKX/XRP/liquidation; pending=1 |
| venue_asset_source | HL/XRP/okx_pressure | 1 | 1 | 1.000 | 0.001225 | 0.001225 | 0.001225 | wait_for_second_label | examples=HL/XRP/okx_pressure; pending=1 |
| venue_asset_source | HL/XRP/liquidation | 1 | 1 | 1.000 | 0.001225 | 0.001225 | 0.001225 | wait_for_second_label | examples=HL/XRP/liquidation; pending=1 |
| venue_asset_source | OKX/TON/liquidation | 1 | 1 | 1.000 | 0.001161 | 0.001161 | 0.001161 | wait_for_second_label | examples=OKX/TON/liquidation; pending=1 |
| venue_asset_source | OKX/LTC/liquidation | 1 | 1 | 1.000 | 0.000950 | 0.000950 | 0.000950 | wait_for_second_label | examples=OKX/LTC/liquidation; pending=1 |
| venue_asset_source | HL/LTC/liquidation | 1 | 1 | 1.000 | 0.000736 | 0.000736 | 0.000736 | wait_for_second_label | examples=HL/LTC/liquidation; pending=1 |
| venue_asset_source | HL/TON/liquidation | 1 | 1 | 1.000 | 0.000696 | 0.000696 | 0.000696 | wait_for_second_label | examples=HL/TON/liquidation; pending=1 |
| asset_source | BTC/liquidation | 1 | 1 | 1.000 | 0.000434 | 0.000434 | 0.000434 | wait_for_second_label | examples=HL/BTC/liquidation; pending=1 |
| venue_asset_source | HL/BTC/liquidation | 1 | 1 | 1.000 | 0.000434 | 0.000434 | 0.000434 | wait_for_second_label | examples=HL/BTC/liquidation; pending=1 |
| asset_source | BTC/l2_imbalance | 1 | 1 | 1.000 | 0.000434 | 0.000434 | 0.000434 | wait_for_second_label | examples=HL/BTC/l2_imbalance; pending=1 |
