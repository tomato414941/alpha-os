# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | SOL | chain_flow_reversal_watch |  | 0.38248296 | -0.38248296 | long_liquidation_cascade_watch | 0.03314302 | 0.637326 | funding helps direction; has recent liquidation context |
| HL | SOL | chain_flow_reversal_watch |  | 0.17846047 | -0.17846047 | long_liquidation_cascade_watch | 0.03314302 | 0.494790 | funding helps direction; has recent liquidation context |
| OKX | HYPE | chain_flow_reversal_watch |  | 0.06815595 | -0.06815595 | short_liquidation_squeeze_watch | 0.06186958 | 0.483903 | funding helps direction; has recent liquidation context |
| OKX | ETH | chain_flow_reversal_watch |  | 0.15704855 | -0.15704855 | mixed_liquidation_flow_watch | 0.02706681 | 0.424649 | funding helps direction; has recent liquidation context |
| OKX | SUI | chain_flow_reversal_watch |  | -0.06464509 | 0.06464509 | long_liquidation_cascade_watch | 0.08614127 | 0.368514 | has recent liquidation context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.06186958 | 0.302624 | has recent liquidation context |
| OKX | BTC | chain_flow_reversal_watch |  | -0.00598260 | 0.00598260 | short_liquidation_squeeze_watch | 0.02886173 | 0.281830 | has recent liquidation context |
| OKX | MON | chain_flow_reversal_watch |  | 0.79807481 | -0.79807481 |  |  | 0.272261 | funding helps direction |
| HL | ETH | chain_flow_reversal_watch |  | -0.05222624 | 0.05222624 | mixed_liquidation_flow_watch | 0.02706681 | 0.188479 | has recent liquidation context |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.08614127 | 0.177881 | has recent liquidation context |
| OKX | NEAR | chain_flow_reversal_watch |  | 0.03880284 | -0.03880284 | short_liquidation_squeeze_watch | 0.03570662 | 0.151430 | funding helps direction; has recent liquidation context |
| HL | BTC | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.02886173 | 0.149943 | has recent liquidation context |
| OKX | MEGA | chain_flow_reversal_watch |  | 0.23267659 | -0.23267659 |  |  | 0.129210 | funding helps direction |
| OKX | AVAX | chain_flow_reversal_watch |  | 0.08609638 | -0.08609638 |  |  | 0.011352 | funding helps direction |
| OKX | BERA | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STX | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | MOVE | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | SEI | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | POL | chain_outflow_stress_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STRK | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | ARB | chain_flow_reversal_watch |  | 0.00848212 | -0.00848212 |  |  | -0.052646 | funding helps direction |
| OKX | APT | chain_flow_reversal_watch |  | -0.02741399 | 0.02741399 |  |  | -0.102147 | weak current context |
| OKX | ADA | chain_flow_reversal_watch |  | 0.16319057 | -0.16319057 |  |  | -0.144029 | funding helps direction |
| OKX | BNB | chain_flow_reversal_watch |  | -0.08573098 | 0.08573098 |  |  | -0.169588 | weak current context |
| HL | ADA | chain_flow_reversal_watch |  | 0.02079274 | -0.02079274 |  |  | -0.187994 | funding helps direction |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.243907 | weak current context |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.03570662 | -0.252434 | has recent liquidation context |
| HL | APT | chain_flow_reversal_watch |  | 0.11547870 | -0.11547870 |  |  | -0.257211 | funding helps direction |
| OKX | TON | chain_flow_reversal_watch |  | -0.03910978 | 0.03910978 | short_liquidation_squeeze_watch | 0.00609642 | -0.267749 | weak current context |
| HL | XLM | chain_outflow_stress_watch |  | 0.10428955 | 0.10428955 |  |  | -0.276421 | funding helps direction |
| HL | AVAX | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.355868 | weak current context |
| HL | ARB | chain_flow_reversal_watch |  | 0.05839854 | -0.05839854 |  |  | -0.365638 | funding helps direction |
| OKX | XLM | chain_outflow_stress_watch |  | -0.13558706 | -0.13558706 |  |  | -0.377191 | weak current context |
| OKX | OP | chain_flow_reversal_watch |  | 0.00611900 | -0.00611900 |  |  | -0.493881 | funding helps direction |
| HL | MOVE | chain_flow_reversal_watch |  | 7.95393896 | -7.95393896 |  |  | -0.500000 | funding helps direction |
| HL | TON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00609642 | -0.521699 | weak current context |
| HL | MEGA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | MON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | OP | chain_flow_reversal_watch |  | 0.38005698 | -0.38005698 |  |  | -0.619943 | funding helps direction |
| HL | SEI | chain_flow_reversal_watch |  | 0.17257988 | -0.17257988 |  |  | -0.807516 | funding helps direction |
