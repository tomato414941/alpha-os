# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | MON | chain_flow_reversal_watch | 0.00641319 | 0.80572183 | -0.80572183 |  |  | 0.912643 | price label positive; funding helps direction |
| OKX | HYPE | chain_flow_reversal_watch | 0.00678311 | 0.06081031 | -0.06081031 | mixed_liquidation_flow_watch | 0.00082611 | 0.663044 | price label positive; funding helps direction |
| OKX | SOL | chain_flow_reversal_watch | 0.00306513 | 0.38634890 | -0.38634890 |  |  | 0.616357 | price label positive; funding helps direction |
| HL | HYPE | chain_flow_reversal_watch | 0.00668993 | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00082611 | 0.559328 | price label positive |
| OKX | ETH | chain_flow_reversal_watch | 0.00191994 | 0.15622280 | -0.15622280 | short_liquidation_squeeze_watch | 0.01962368 | 0.541390 | price label positive; funding helps direction; has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch | 0.00165604 | 0.13897127 | -0.13897127 | short_liquidation_squeeze_watch | 0.01962368 | 0.448743 | price label positive; funding helps direction; has recent liquidation context |
| HL | SOL | chain_flow_reversal_watch | 0.00295654 | 0.15540240 | -0.15540240 |  |  | 0.443409 | price label positive; funding helps direction |
| OKX | KAT | chain_flow_reversal_watch | 0.00290867 |  |  |  |  | 0.290867 | price label positive |
| OKX | POL | chain_flow_reversal_watch | 0.00253036 |  |  |  |  | 0.253036 | price label positive |
| OKX | MEGA | chain_flow_reversal_watch | 0.00020670 | 0.26241290 | -0.26241290 |  |  | 0.180361 | price label positive; funding helps direction |
| OKX | MOVE | chain_flow_reversal_watch | 0.00170648 |  |  |  |  | 0.170648 | price label positive |
| OKX | BERA | chain_flow_reversal_watch | 0.00164813 |  |  |  |  | 0.164813 | price label positive |
| OKX | BTC | chain_flow_reversal_watch | 0.00141498 | -0.00963477 | 0.00963477 | short_liquidation_squeeze_watch | 0.00059717 | 0.137031 | price label positive |
| OKX | STX | chain_flow_reversal_watch | 0.00107875 |  |  |  |  | 0.107875 | price label positive |
| OKX | SEI | chain_flow_reversal_watch | 0.00101688 |  |  |  |  | 0.101688 | price label positive |
| OKX | APT | chain_flow_reversal_watch | 0.00194378 | -0.01920559 | 0.01920559 |  |  | 0.100607 | price label positive |
| OKX | BNB | chain_flow_reversal_watch | 0.00218194 | -0.07965044 | 0.07965044 |  |  | 0.054770 | price label positive |
| OKX | ARB | chain_flow_reversal_watch | 0.00110011 | 0.00013278 | -0.00013278 |  |  | 0.049120 | price label positive; funding helps direction |
| HL | MON | chain_flow_reversal_watch | 0.00557129 | -0.10950000 | 0.10950000 |  |  | 0.036989 | price label positive |
| HL | BTC | chain_flow_reversal_watch | 0.00141620 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00059717 | 0.030054 | price label positive |
| OKX | ADA | chain_flow_reversal_watch | 0.00122850 | 0.16470644 | -0.16470644 | short_liquidation_squeeze_watch | 0.00044604 | -0.014262 | price label positive; funding helps direction |
| OKX | TON | chain_flow_reversal_watch | -0.00115407 | -0.03393936 | 0.03393936 | short_liquidation_squeeze_watch | 0.04191457 | -0.018801 | has recent liquidation context |
| HL | BNB | chain_flow_reversal_watch | 0.00172830 | -0.10950000 | 0.10950000 |  |  | -0.042184 | price label positive |
| OKX | OP | chain_flow_reversal_watch | 0.00420610 | 0.01300102 | -0.01300102 |  |  | -0.066389 | price label positive; funding helps direction |
| OKX | SUI | chain_flow_reversal_watch | 0.00040021 | -0.06427009 | 0.06427009 | short_liquidation_squeeze_watch | 0.00155074 | -0.075448 | price label positive |
| HL | ADA | chain_flow_reversal_watch | 0.00184196 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00044604 | -0.132021 | price label positive |
| HL | SUI | chain_flow_reversal_watch | 0.00038650 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00155074 | -0.168647 | price label positive |
| HL | TON | chain_flow_reversal_watch | -0.00126765 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.04191457 | -0.246626 | has recent liquidation context |
| HL | SEI | chain_flow_reversal_watch | 0.00191224 | 0.74291020 | -0.74291020 |  |  | -0.275535 | price label positive; funding helps direction |
| HL | APT | chain_flow_reversal_watch | 0.00179399 | -0.10950000 | 0.10950000 |  |  | -0.302790 | price label positive |
| OKX | STRK | chain_outflow_stress_watch | -0.00360469 |  |  |  |  | -0.360469 | weak current context |
| HL | AVAX | chain_outflow_stress_watch | -0.00302282 | 0.06122101 | 0.06122101 |  |  | -0.412567 | funding helps direction |
| HL | ARB | chain_flow_reversal_watch | 0.00134491 | -0.10950000 | 0.10950000 |  |  | -0.475009 | price label positive |
| HL | MEGA | chain_flow_reversal_watch | -0.00039289 | 0.01124784 | -0.01124784 |  |  | -0.528041 | funding helps direction |
| OKX | AVAX | chain_outflow_stress_watch | -0.00419099 | -0.09662321 | -0.09662321 |  |  | -0.590299 | weak current context |
| OKX | NEAR | chain_outflow_stress_watch | -0.00442043 | -0.03291537 | -0.03291537 | long_liquidation_cascade_watch | 0.00985181 | -0.620641 | weak current context |
| HL | OP | chain_flow_reversal_watch | 0.00441223 | -0.09016931 | 0.09016931 |  |  | -0.648946 | price label positive |
| HL | NEAR | chain_outflow_stress_watch | -0.00486152 | 0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00985181 | -0.675937 | funding helps direction |
| HL | POL | chain_flow_reversal_watch | 0.00223878 | -0.01380488 | 0.01380488 |  |  | -0.687408 | price label positive |
| HL | STX | chain_flow_reversal_watch | 0.00059331 | 0.07494968 | -0.07494968 |  |  | -0.865719 | price label positive; funding helps direction |
