# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | ETH | chain_flow_reversal_watch |  | 0.07598815 | -0.07598815 | short_liquidation_squeeze_watch | 0.06805380 | 0.573043 | funding helps direction; has recent liquidation context |
| OKX | BTC | chain_flow_reversal_watch |  | 0.08956514 | -0.08956514 | short_liquidation_squeeze_watch | 0.04657199 | 0.551381 | funding helps direction; has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.06805380 | 0.328635 | has recent liquidation context |
| HL | BTC | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.04657199 | 0.288296 | has recent liquidation context |
| OKX | MEGA | chain_flow_reversal_watch |  | 0.36992183 | -0.36992183 |  |  | 0.271254 | funding helps direction |
| OKX | BNB | chain_flow_reversal_watch |  | 0.04088892 | -0.04088892 | short_liquidation_squeeze_watch | 0.02936442 | 0.251800 | funding helps direction; has recent liquidation context |
| OKX | MOVE | chain_flow_reversal_watch |  | 1.71372964 | -1.71372964 |  |  | 0.143748 | funding helps direction |
| OKX | SOL | chain_flow_reversal_watch |  | 0.06482913 | -0.06482913 | short_liquidation_squeeze_watch | 0.00864619 | 0.077157 | funding helps direction |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.02936442 | 0.064290 | has recent liquidation context |
| OKX | BERA | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STX | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| HL | MOVE | chain_flow_reversal_watch |  | 1.73576772 | -1.73576772 |  |  | 0.000000 | funding helps direction |
| OKX | SEI | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | POL | chain_outflow_stress_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STRK | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | HYPE | chain_flow_reversal_watch |  | -0.03093882 | 0.03093882 | long_liquidation_cascade_watch | 0.01073090 | -0.001071 | has recent liquidation context |
| HL | SOL | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00864619 | -0.044520 | weak current context |
| OKX | XLM | chain_outflow_stress_watch |  | -0.02129203 | -0.02129203 |  |  | -0.046076 | weak current context |
| OKX | ARB | chain_flow_reversal_watch |  | 0.01332677 | -0.01332677 |  |  | -0.046109 | funding helps direction |
| OKX | AVAX | chain_flow_reversal_watch |  | 0.00697006 | -0.00697006 |  |  | -0.066199 | funding helps direction |
| HL | APT | chain_flow_reversal_watch |  | 0.36992779 | -0.36992779 |  |  | -0.071314 | funding helps direction |
| HL | ADA | chain_flow_reversal_watch |  | 0.10557552 | -0.10557552 | short_liquidation_squeeze_watch | 0.00085733 | -0.071462 | funding helps direction |
| OKX | SUI | chain_flow_reversal_watch |  | -0.00739881 | 0.00739881 |  |  | -0.072601 | weak current context |
| OKX | NEAR | chain_flow_reversal_watch |  | 0.11291075 | -0.11291075 | short_liquidation_squeeze_watch | 0.00153694 | -0.101658 | funding helps direction |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.115994 | weak current context |
| OKX | APT | chain_flow_reversal_watch |  | -0.05475801 | 0.05475801 |  |  | -0.128336 | weak current context |
| HL | AVAX | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.175254 | weak current context |
| OKX | TON | chain_flow_reversal_watch |  | -0.03456519 | 0.03456519 | short_liquidation_squeeze_watch | 0.01313689 | -0.191464 | has recent liquidation context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.28021050 | 0.28021050 | long_liquidation_cascade_watch | 0.01073090 | -0.229961 | has recent liquidation context |
| OKX | MON | chain_flow_reversal_watch |  | -0.05475000 | 0.05475000 |  |  | -0.278815 | weak current context |
| HL | XLM | chain_outflow_stress_watch |  | 0.08473460 | 0.08473460 |  |  | -0.345713 | funding helps direction |
| OKX | ADA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00085733 | -0.397048 | weak current context |
| HL | TON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.01313689 | -0.398167 | has recent liquidation context |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00153694 | -0.423748 | weak current context |
| HL | MEGA | chain_flow_reversal_watch |  | -0.05181014 | 0.05181014 |  |  | -0.551810 | weak current context |
| HL | ARB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.584559 | weak current context |
| OKX | OP | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | MON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | SEI | chain_flow_reversal_watch |  | 0.13603842 | -0.13603842 |  |  | -0.863962 | funding helps direction |
| HL | STX | chain_flow_reversal_watch |  | 0.11219720 | -0.11219720 |  |  | -0.887803 | funding helps direction |
