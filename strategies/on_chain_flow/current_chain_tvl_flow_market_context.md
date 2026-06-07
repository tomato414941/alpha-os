# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | ETH | chain_flow_reversal_watch | 0.00088840 | 0.15704855 | -0.15704855 | short_liquidation_squeeze_watch | 0.01030080 | 0.345829 | price label positive; funding helps direction; has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch | 0.00091906 | 0.18074245 | -0.18074245 | short_liquidation_squeeze_watch | 0.01030080 | 0.344977 | price label positive; funding helps direction; has recent liquidation context |
| OKX | XLM | chain_flow_reversal_watch | 0.00241779 | 0.13558706 | -0.13558706 | long_liquidation_cascade_watch | 0.00031521 | 0.138914 | price label positive; funding helps direction |
| OKX | POL | chain_flow_reversal_watch | 0.00126247 |  |  |  |  | 0.126247 | price label positive |
| OKX | SOL | chain_flow_reversal_watch | -0.00214198 | 0.38248296 | -0.38248296 |  |  | 0.091698 | funding helps direction |
| OKX | MEGA | chain_flow_reversal_watch | -0.00145349 | 0.23267659 | -0.23267659 |  |  | -0.016139 | funding helps direction |
| HL | SOL | chain_flow_reversal_watch | -0.00206473 | 0.17098206 | -0.17098206 |  |  | -0.043147 | funding helps direction |
| HL | AVAX | chain_outflow_stress_watch | 0.00095665 | 0.00289868 | 0.00289868 |  |  | -0.043414 | price label positive; funding helps direction |
| OKX | AVAX | chain_outflow_stress_watch | 0.00104603 | -0.08609638 | -0.08609638 |  |  | -0.056237 | price label positive |
| OKX | STRK | chain_outflow_stress_watch | -0.00059916 |  |  |  |  | -0.059916 | weak current context |
| OKX | BTC | chain_flow_reversal_watch | -0.00069463 | -0.00598260 | 0.00598260 | short_liquidation_squeeze_watch | 0.00047763 | -0.071474 | weak current context |
| OKX | MOVE | chain_flow_reversal_watch | -0.00085251 |  |  |  |  | -0.085251 | weak current context |
| OKX | SUI | chain_flow_reversal_watch | 0.00026731 | -0.06464509 | 0.06464509 | short_liquidation_squeeze_watch | 0.00155231 | -0.089231 | price label positive |
| OKX | ARB | chain_flow_reversal_watch | -0.00110227 | 0.00848212 | -0.00848212 |  |  | -0.162873 | funding helps direction |
| OKX | STX | chain_flow_reversal_watch | -0.00216216 |  |  |  |  | -0.216216 | weak current context |
| OKX | ADA | chain_flow_reversal_watch | -0.00123229 | 0.16319057 | -0.16319057 | short_liquidation_squeeze_watch | 0.00044734 | -0.262785 | funding helps direction |
| HL | SUI | chain_flow_reversal_watch | 0.00028040 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00155231 | -0.266290 | price label positive |
| HL | BTC | chain_flow_reversal_watch | -0.00083585 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00047763 | -0.279211 | weak current context |
| OKX | BERA | chain_flow_reversal_watch | -0.00289256 |  |  |  |  | -0.289256 | weak current context |
| OKX | KAT | chain_flow_reversal_watch | -0.00311405 |  |  |  |  | -0.311405 | weak current context |
| OKX | SEI | chain_flow_reversal_watch | -0.00325402 |  |  |  |  | -0.325402 | weak current context |
| OKX | BNB | chain_flow_reversal_watch | -0.00167476 | -0.08573098 | 0.08573098 |  |  | -0.337064 | weak current context |
| OKX | HYPE | chain_flow_reversal_watch | -0.00336757 | 0.06815595 | -0.06815595 | mixed_liquidation_flow_watch | 0.00146531 | -0.338201 | funding helps direction |
| OKX | APT | chain_flow_reversal_watch | -0.00299133 | -0.02741399 | 0.02741399 |  |  | -0.401280 | weak current context |
| HL | BNB | chain_flow_reversal_watch | -0.00162433 | -0.10950000 | 0.10950000 |  |  | -0.410227 | weak current context |
| HL | XLM | chain_flow_reversal_watch | 0.00188415 | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00031521 | -0.413138 | price label positive |
| HL | ADA | chain_flow_reversal_watch | -0.00116945 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00044734 | -0.449168 | weak current context |
| HL | HYPE | chain_flow_reversal_watch | -0.00316221 | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00146531 | -0.477540 | weak current context |
| HL | POL | chain_flow_reversal_watch | 0.00348621 | -0.01133018 | 0.01133018 |  |  | -0.662709 | price label positive |
| HL | MEGA | chain_flow_reversal_watch | -0.00217779 | 0.03782743 | -0.03782743 |  |  | -0.679952 | funding helps direction |
| HL | ARB | chain_flow_reversal_watch | -0.00110267 | -0.10950000 | 0.10950000 |  |  | -0.719767 | weak current context |
| HL | SEI | chain_flow_reversal_watch | -0.00248129 | 0.71422207 | -0.71422207 |  |  | -0.748129 | funding helps direction |
| OKX | OP | chain_flow_reversal_watch | -0.00315789 | 0.00611900 | -0.00611900 |  |  | -0.809670 | funding helps direction |
| HL | APT | chain_flow_reversal_watch | -0.00299043 | -0.10950000 | 0.10950000 |  |  | -0.856839 | weak current context |
| HL | STX | chain_flow_reversal_watch | -0.00124271 | 0.08127002 | -0.08127002 |  |  | -1.043001 | funding helps direction |
| HL | MOVE | chain_flow_reversal_watch | -0.00008530 | -0.10950000 | 0.10950000 |  |  | -1.118030 | weak current context |
| HL | STRK | chain_outflow_stress_watch | -0.00000000 | -0.12910050 | -0.12910050 |  |  | -1.129101 | weak current context |
| OKX | MON | chain_flow_reversal_watch | -0.01454545 | 0.79807481 | -0.79807481 |  |  | -1.182284 | funding helps direction |
| OKX | NEAR | chain_flow_reversal_watch | -0.01115963 | 0.03880284 | -0.03880284 | long_liquidation_cascade_watch | 0.00983723 | -1.223227 | funding helps direction |
| HL | OP | chain_flow_reversal_watch | -0.00221053 | -0.07523526 | 0.07523526 |  |  | -1.296288 | weak current context |
