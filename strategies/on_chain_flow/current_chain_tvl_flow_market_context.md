# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | ETH | chain_flow_reversal_watch |  | 0.15704855 | -0.15704855 | short_liquidation_squeeze_watch | 0.02754692 | 0.429450 | funding helps direction; has recent liquidation context |
| OKX | SOL | chain_flow_reversal_watch |  | 0.38248296 | -0.38248296 |  |  | 0.305896 | funding helps direction |
| OKX | MON | chain_flow_reversal_watch |  | 0.79807481 | -0.79807481 |  |  | 0.272261 | funding helps direction |
| OKX | BTC | chain_flow_reversal_watch |  | -0.00598260 | 0.00598260 | short_liquidation_squeeze_watch | 0.02457470 | 0.238960 | has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch |  | -0.06873709 | 0.06873709 | short_liquidation_squeeze_watch | 0.02754692 | 0.177073 | has recent liquidation context |
| HL | BTC | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.02457470 | 0.111728 | has recent liquidation context |
| HL | SOL | chain_flow_reversal_watch |  | 0.05121359 | -0.05121359 |  |  | 0.020958 | funding helps direction |
| OKX | MOVE | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | BERA | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STX | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | SEI | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | POL | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STRK | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | HYPE | chain_flow_reversal_watch |  | 0.06815595 | -0.06815595 | long_liquidation_cascade_watch | 0.00032885 | -0.012809 | funding helps direction |
| OKX | ARB | chain_flow_reversal_watch |  | 0.00848212 | -0.00848212 |  |  | -0.052646 | funding helps direction |
| OKX | APT | chain_flow_reversal_watch |  | -0.02741399 | 0.02741399 |  |  | -0.102147 | weak current context |
| OKX | SUI | chain_flow_reversal_watch |  | -0.06464509 | 0.06464509 | mixed_liquidation_flow_watch | 0.00284028 | -0.103083 | weak current context |
| OKX | ADA | chain_flow_reversal_watch |  | 0.16319057 | -0.16319057 |  |  | -0.144029 | funding helps direction |
| OKX | AVAX | chain_outflow_stress_watch |  | -0.08609638 | -0.08609638 |  |  | -0.160840 | weak current context |
| OKX | BNB | chain_flow_reversal_watch |  | -0.08573098 | 0.08573098 |  |  | -0.169588 | weak current context |
| HL | AVAX | chain_outflow_stress_watch |  | 0.00531469 | 0.00531469 |  |  | -0.201269 | funding helps direction |
| OKX | NEAR | chain_flow_reversal_watch |  | 0.03880284 | -0.03880284 |  |  | -0.205636 | funding helps direction |
| HL | ADA | chain_flow_reversal_watch |  | 0.03999203 | -0.03999203 |  |  | -0.212718 | funding helps direction |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.218754 | weak current context |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00284028 | -0.226477 | weak current context |
| OKX | MEGA | chain_outflow_stress_watch |  | -0.23267659 | -0.23267659 |  |  | -0.336143 | weak current context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00032885 | -0.363300 | weak current context |
| OKX | XLM | chain_outflow_stress_watch |  | -0.13558706 | -0.13558706 |  |  | -0.377191 | weak current context |
| HL | MEGA | chain_outflow_stress_watch |  | 0.10950000 | 0.10950000 |  |  | -0.390500 | funding helps direction |
| HL | XLM | chain_outflow_stress_watch |  | 0.10950000 | 0.10950000 |  |  | -0.390500 | funding helps direction |
| OKX | OP | chain_flow_reversal_watch |  | 0.00611900 | -0.00611900 |  |  | -0.493881 | funding helps direction |
| HL | MON | chain_flow_reversal_watch |  | 0.00359948 | -0.00359948 |  |  | -0.496401 | funding helps direction |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.568126 | weak current context |
| HL | ARB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | APT | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | MOVE | chain_flow_reversal_watch |  | 0.23322186 | -0.23322186 |  |  | -0.766778 | funding helps direction |
| HL | POL | chain_flow_reversal_watch |  | 0.15000098 | -0.15000098 |  |  | -0.849999 | funding helps direction |
| HL | STRK | chain_flow_reversal_watch |  | 0.05565228 | -0.05565228 |  |  | -0.890511 | funding helps direction |
| HL | SEI | chain_flow_reversal_watch |  | 0.03598871 | -0.03598871 |  |  | -0.964011 | funding helps direction |
| HL | MNT | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -1.013800 | weak current context |
