# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | SOL | chain_flow_reversal_watch |  | 0.38248296 | -0.38248296 | long_liquidation_cascade_watch | 0.03314302 | 0.637326 | funding helps direction; has recent liquidation context |
| OKX | HYPE | chain_flow_reversal_watch |  | 0.06815595 | -0.06815595 | short_liquidation_squeeze_watch | 0.06186958 | 0.483903 | funding helps direction; has recent liquidation context |
| OKX | ETH | chain_flow_reversal_watch |  | 0.15704855 | -0.15704855 | mixed_liquidation_flow_watch | 0.02706681 | 0.424649 | funding helps direction; has recent liquidation context |
| OKX | SUI | chain_flow_reversal_watch |  | -0.06464509 | 0.06464509 | long_liquidation_cascade_watch | 0.08614127 | 0.368514 | has recent liquidation context |
| OKX | BTC | chain_flow_reversal_watch |  | -0.00598260 | 0.00598260 | short_liquidation_squeeze_watch | 0.02886173 | 0.281830 | has recent liquidation context |
| OKX | MON | chain_flow_reversal_watch |  | 0.79807481 | -0.79807481 |  |  | 0.272261 | funding helps direction |
| HL | SOL | chain_flow_reversal_watch |  | 0.11745758 | -0.11745758 | long_liquidation_cascade_watch | 0.03314302 | 0.263877 | funding helps direction; has recent liquidation context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.06186958 | 0.226298 | has recent liquidation context |
| HL | BTC | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.02886173 | 0.171235 | has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch |  | -0.03407728 | 0.03407728 | mixed_liquidation_flow_watch | 0.02706681 | 0.158564 | has recent liquidation context |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.08614127 | 0.158016 | has recent liquidation context |
| OKX | NEAR | chain_flow_reversal_watch |  | 0.03880284 | -0.03880284 | short_liquidation_squeeze_watch | 0.03570662 | 0.151430 | funding helps direction; has recent liquidation context |
| OKX | MEGA | chain_flow_reversal_watch |  | 0.23267659 | -0.23267659 |  |  | 0.129210 | funding helps direction |
| OKX | AVAX | chain_flow_reversal_watch |  | 0.08609638 | -0.08609638 |  |  | 0.011352 | funding helps direction |
| OKX | BERA | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STX | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | MOVE | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | SEI | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | POL | chain_outflow_stress_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STRK | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | ARB | chain_flow_reversal_watch |  | 0.00848212 | -0.00848212 |  |  | -0.052646 | funding helps direction |
| HL | APT | chain_flow_reversal_watch |  | 0.38630636 | -0.38630636 |  |  | -0.063536 | funding helps direction |
| OKX | APT | chain_flow_reversal_watch |  | -0.02741399 | 0.02741399 |  |  | -0.102147 | weak current context |
| OKX | ADA | chain_flow_reversal_watch |  | 0.16319057 | -0.16319057 |  |  | -0.144029 | funding helps direction |
| OKX | BNB | chain_flow_reversal_watch |  | -0.08573098 | 0.08573098 |  |  | -0.169588 | weak current context |
| HL | AVAX | chain_flow_reversal_watch |  | 0.02883442 | -0.02883442 |  |  | -0.233177 | funding helps direction |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.03570662 | -0.241414 | has recent liquidation context |
| HL | ADA | chain_flow_reversal_watch |  | 0.06273474 | -0.06273474 |  |  | -0.266122 | funding helps direction |
| OKX | TON | chain_flow_reversal_watch |  | -0.03910978 | 0.03910978 | short_liquidation_squeeze_watch | 0.00609642 | -0.267749 | weak current context |
| HL | ARB | chain_flow_reversal_watch |  | 0.17062816 | -0.17062816 |  |  | -0.329372 | funding helps direction |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.366614 | weak current context |
| OKX | XLM | chain_outflow_stress_watch |  | -0.13558706 | -0.13558706 |  |  | -0.377191 | weak current context |
| HL | XLM | chain_outflow_stress_watch |  | 0.10950000 | 0.10950000 |  |  | -0.390500 | funding helps direction |
| HL | MEGA | chain_flow_reversal_watch |  | 0.08287398 | -0.08287398 |  |  | -0.417126 | funding helps direction |
| HL | MON | chain_flow_reversal_watch |  | 0.06001126 | -0.06001126 |  |  | -0.439989 | funding helps direction |
| OKX | OP | chain_flow_reversal_watch |  | 0.00611900 | -0.00611900 |  |  | -0.493881 | funding helps direction |
| HL | MOVE | chain_flow_reversal_watch |  | 10.92133728 | -10.92133728 |  |  | -0.500000 | funding helps direction |
| HL | TON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00609642 | -0.548536 | weak current context |
| HL | MNT | chain_outflow_stress_watch |  | 0.10950000 | 0.10950000 |  |  | -0.890500 | funding helps direction |
| HL | POL | chain_outflow_stress_watch |  | 0.10950000 | 0.10950000 |  |  | -0.890500 | funding helps direction |
