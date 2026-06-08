# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | BTC | chain_flow_reversal_watch |  | 0.09020081 | -0.09020081 | short_liquidation_squeeze_watch | 0.08847140 | 0.589421 | funding helps direction; has recent liquidation context |
| OKX | ETH | chain_flow_reversal_watch |  | 0.07515203 | -0.07515203 | short_liquidation_squeeze_watch | 0.11883760 | 0.572215 | funding helps direction; has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch |  | 0.13751623 | -0.13751623 | short_liquidation_squeeze_watch | 0.11883760 | 0.569358 | funding helps direction; has recent liquidation context |
| HL | BTC | chain_flow_reversal_watch |  | -0.10599600 | 0.10599600 | short_liquidation_squeeze_watch | 0.08847140 | 0.386129 | has recent liquidation context |
| OKX | BNB | chain_flow_reversal_watch |  | 0.03606454 | -0.03606454 | short_liquidation_squeeze_watch | 0.03516253 | 0.305134 | funding helps direction; has recent liquidation context |
| OKX | MEGA | chain_flow_reversal_watch |  | 0.35674837 | -0.35674837 |  |  | 0.259025 | funding helps direction |
| HL | ADA | chain_flow_reversal_watch |  | 0.32782811 | -0.32782811 | short_liquidation_squeeze_watch | 0.01129842 | 0.193374 | funding helps direction; has recent liquidation context |
| OKX | SOL | chain_flow_reversal_watch |  | 0.05507726 | -0.05507726 | short_liquidation_squeeze_watch | 0.01679511 | 0.149157 | funding helps direction; has recent liquidation context |
| OKX | MOVE | chain_flow_reversal_watch |  | 1.44892472 | -1.44892472 |  |  | 0.144508 | funding helps direction |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.03516253 | 0.144251 | has recent liquidation context |
| HL | SOL | chain_flow_reversal_watch |  | -0.02648761 | 0.02648761 | short_liquidation_squeeze_watch | 0.01679511 | 0.131088 | has recent liquidation context |
| OKX | SUI | chain_flow_reversal_watch |  | -0.00502467 | 0.00502467 | short_liquidation_squeeze_watch | 0.00949015 | 0.025064 | weak current context |
| OKX | STX | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | BERA | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| HL | MOVE | chain_flow_reversal_watch |  | 1.01632732 | -1.01632732 |  |  | 0.000000 | funding helps direction |
| OKX | SEI | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | POL | chain_outflow_stress_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STRK | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.01068447 | -0.010528 | has recent liquidation context |
| OKX | HYPE | chain_flow_reversal_watch |  | -0.05114811 | 0.05114811 | long_liquidation_cascade_watch | 0.01068447 | -0.021351 | has recent liquidation context |
| OKX | ARB | chain_flow_reversal_watch |  | 0.01634771 | -0.01634771 |  |  | -0.042876 | funding helps direction |
| OKX | XLM | chain_outflow_stress_watch |  | -0.03579400 | -0.03579400 |  |  | -0.060407 | weak current context |
| OKX | AVAX | chain_flow_reversal_watch |  | 0.00459786 | -0.00459786 |  |  | -0.068198 | funding helps direction |
| OKX | TON | chain_flow_reversal_watch |  | -0.03195771 | 0.03195771 | short_liquidation_squeeze_watch | 0.02463480 | -0.071079 | has recent liquidation context |
| OKX | NEAR | chain_flow_reversal_watch |  | 0.11906106 | -0.11906106 | short_liquidation_squeeze_watch | 0.00153514 | -0.094578 | funding helps direction |
| OKX | APT | chain_flow_reversal_watch |  | -0.05050262 | 0.05050262 |  |  | -0.123822 | weak current context |
| HL | AVAX | chain_flow_reversal_watch |  | 0.03632334 | -0.03632334 |  |  | -0.124880 | funding helps direction |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00949015 | -0.205185 | weak current context |
| HL | TON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.02463480 | -0.233751 | has recent liquidation context |
| HL | XLM | chain_outflow_stress_watch |  | 0.02983656 | 0.02983656 |  |  | -0.276942 | funding helps direction |
| OKX | MON | chain_flow_reversal_watch |  | -0.05475000 | 0.05475000 |  |  | -0.278214 | weak current context |
| OKX | ADA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.01129842 | -0.289513 | has recent liquidation context |
| HL | ARB | chain_flow_reversal_watch |  | 0.04212246 | -0.04212246 |  |  | -0.457878 | funding helps direction |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00153514 | -0.458821 | weak current context |
| HL | APT | chain_flow_reversal_watch |  | 0.03017119 | -0.03017119 |  |  | -0.469829 | funding helps direction |
| HL | MEGA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| OKX | OP | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | MON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | POL | chain_outflow_stress_watch |  | 0.10950000 | 0.10950000 |  |  | -0.890500 | funding helps direction |
| HL | SEI | chain_flow_reversal_watch |  | 0.09122927 | -0.09122927 |  |  | -0.908771 | funding helps direction |
