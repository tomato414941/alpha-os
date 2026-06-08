# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | ETH | chain_flow_reversal_watch |  | 0.15704855 | -0.15704855 | short_liquidation_squeeze_watch | 0.02754692 | 0.429450 | funding helps direction; has recent liquidation context |
| OKX | SOL | chain_flow_reversal_watch |  | 0.38248296 | -0.38248296 |  |  | 0.305896 | funding helps direction |
| OKX | MON | chain_flow_reversal_watch |  | 0.79807481 | -0.79807481 |  |  | 0.272261 | funding helps direction |
| OKX | BTC | chain_flow_reversal_watch |  | -0.00598260 | 0.00598260 | short_liquidation_squeeze_watch | 0.02457470 | 0.238960 | has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.02754692 | 0.124459 | has recent liquidation context |
| HL | BTC | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.02457470 | 0.111739 | has recent liquidation context |
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
| HL | SOL | chain_flow_reversal_watch |  | -0.02392882 | 0.02392882 |  |  | -0.106768 | weak current context |
| HL | AVAX | chain_outflow_stress_watch |  | 0.10950000 | 0.10950000 |  |  | -0.132805 | funding helps direction |
| OKX | ADA | chain_flow_reversal_watch |  | 0.16319057 | -0.16319057 |  |  | -0.144029 | funding helps direction |
| OKX | AVAX | chain_outflow_stress_watch |  | -0.08609638 | -0.08609638 |  |  | -0.160840 | weak current context |
| OKX | BNB | chain_flow_reversal_watch |  | -0.08573098 | 0.08573098 |  |  | -0.169588 | weak current context |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.183961 | weak current context |
| OKX | NEAR | chain_flow_reversal_watch |  | 0.03880284 | -0.03880284 |  |  | -0.205636 | funding helps direction |
| HL | ADA | chain_flow_reversal_watch |  | -0.09442930 | 0.09442930 |  |  | -0.239796 | weak current context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00032885 | -0.287881 | weak current context |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00284028 | -0.298623 | weak current context |
| OKX | MEGA | chain_outflow_stress_watch |  | -0.23267659 | -0.23267659 |  |  | -0.336143 | weak current context |
| OKX | XLM | chain_outflow_stress_watch |  | -0.13558706 | -0.13558706 |  |  | -0.377191 | weak current context |
| HL | XLM | chain_outflow_stress_watch |  | 0.10950000 | 0.10950000 |  |  | -0.390500 | funding helps direction |
| OKX | OP | chain_flow_reversal_watch |  | 0.00611900 | -0.00611900 |  |  | -0.493881 | funding helps direction |
| HL | MON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.538665 | weak current context |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.549921 | weak current context |
| HL | MEGA | chain_outflow_stress_watch |  | -0.10783210 | -0.10783210 |  |  | -0.607832 | weak current context |
| HL | ARB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | APT | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | POL | chain_flow_reversal_watch |  | 0.13388434 | -0.13388434 |  |  | -0.856293 | funding helps direction |
| HL | SEI | chain_flow_reversal_watch |  | 0.05478416 | -0.05478416 |  |  | -0.945216 | funding helps direction |
| HL | STX | chain_flow_reversal_watch |  | -0.05553752 | 0.05553752 |  |  | -1.055538 | weak current context |
| HL | MOVE | chain_flow_reversal_watch |  | -0.05697592 | 0.05697592 |  |  | -1.056976 | weak current context |
| HL | MNT | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -1.109500 | weak current context |
