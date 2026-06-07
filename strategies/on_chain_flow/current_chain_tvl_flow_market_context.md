# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | SOL | chain_flow_reversal_watch |  | 0.38248296 | -0.38248296 |  |  | 0.305896 | funding helps direction |
| OKX | MON | chain_flow_reversal_watch |  | 0.79807481 | -0.79807481 |  |  | 0.272261 | funding helps direction |
| OKX | ETH | chain_flow_reversal_watch |  | 0.15704855 | -0.15704855 | short_liquidation_squeeze_watch | 0.01030080 | 0.256989 | funding helps direction; has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch |  | 0.18074245 | -0.18074245 | short_liquidation_squeeze_watch | 0.01030080 | 0.253071 | funding helps direction; has recent liquidation context |
| HL | SOL | chain_flow_reversal_watch |  | 0.17098206 | -0.17098206 |  |  | 0.163326 | funding helps direction |
| OKX | MEGA | chain_flow_reversal_watch |  | 0.23267659 | -0.23267659 |  |  | 0.129210 | funding helps direction |
| OKX | KAT | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | MOVE | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | BERA | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STX | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | SEI | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | POL | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STRK | chain_outflow_stress_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | HYPE | chain_flow_reversal_watch |  | 0.06815595 | -0.06815595 | mixed_liquidation_flow_watch | 0.00146531 | -0.001444 | funding helps direction |
| OKX | BTC | chain_flow_reversal_watch |  | -0.00598260 | 0.00598260 | short_liquidation_squeeze_watch | 0.00047763 | -0.002011 | weak current context |
| OKX | ARB | chain_flow_reversal_watch |  | 0.00848212 | -0.00848212 |  |  | -0.052646 | funding helps direction |
| OKX | APT | chain_flow_reversal_watch |  | -0.02741399 | 0.02741399 |  |  | -0.102147 | weak current context |
| OKX | XLM | chain_flow_reversal_watch |  | 0.13558706 | -0.13558706 | long_liquidation_cascade_watch | 0.00031521 | -0.102865 | funding helps direction |
| OKX | NEAR | chain_flow_reversal_watch |  | 0.03880284 | -0.03880284 | long_liquidation_cascade_watch | 0.00983723 | -0.107264 | funding helps direction |
| OKX | SUI | chain_flow_reversal_watch |  | -0.06464509 | 0.06464509 | short_liquidation_squeeze_watch | 0.00155231 | -0.115962 | weak current context |
| HL | AVAX | chain_outflow_stress_watch |  | 0.00289868 | 0.00289868 |  |  | -0.139079 | funding helps direction |
| OKX | ADA | chain_flow_reversal_watch |  | 0.16319057 | -0.16319057 | short_liquidation_squeeze_watch | 0.00044734 | -0.139556 | funding helps direction |
| OKX | AVAX | chain_outflow_stress_watch |  | -0.08609638 | -0.08609638 |  |  | -0.160840 | weak current context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00146531 | -0.161319 | weak current context |
| OKX | BNB | chain_flow_reversal_watch |  | -0.08573098 | 0.08573098 |  |  | -0.169588 | weak current context |
| HL | BTC | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00047763 | -0.195626 | weak current context |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.247794 | weak current context |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00155231 | -0.294330 | weak current context |
| HL | ADA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00044734 | -0.332223 | weak current context |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00983723 | -0.458418 | weak current context |
| HL | MEGA | chain_flow_reversal_watch |  | 0.03782743 | -0.03782743 |  |  | -0.462173 | funding helps direction |
| OKX | OP | chain_flow_reversal_watch |  | 0.00611900 | -0.00611900 |  |  | -0.493881 | funding helps direction |
| HL | SEI | chain_flow_reversal_watch |  | 0.71422207 | -0.71422207 |  |  | -0.500000 | funding helps direction |
| HL | MON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.518424 | weak current context |
| HL | APT | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.557796 | weak current context |
| HL | XLM | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00031521 | -0.601553 | weak current context |
| HL | ARB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | STX | chain_flow_reversal_watch |  | 0.08127002 | -0.08127002 |  |  | -0.918730 | funding helps direction |
| HL | POL | chain_flow_reversal_watch |  | -0.01133018 | 0.01133018 |  |  | -1.011330 | weak current context |
| HL | OP | chain_flow_reversal_watch |  | -0.07523526 | 0.07523526 |  |  | -1.075235 | weak current context |
