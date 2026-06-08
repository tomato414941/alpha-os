# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | ETH | chain_flow_reversal_watch |  | 0.07914734 | -0.07914734 | long_liquidation_cascade_watch | 0.04805326 | 0.556703 | funding helps direction; has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch |  | -0.07801306 | 0.07801306 | long_liquidation_cascade_watch | 0.04805326 | 0.349121 | has recent liquidation context |
| OKX | HYPE | chain_flow_reversal_watch |  | 0.14422721 | -0.14422721 | long_liquidation_cascade_watch | 0.02841193 | 0.347551 | funding helps direction; has recent liquidation context |
| OKX | BTC | chain_flow_reversal_watch |  | 0.05241027 | -0.05241027 | mixed_liquidation_flow_watch | 0.02636978 | 0.315317 | funding helps direction; has recent liquidation context |
| OKX | TON | chain_flow_reversal_watch |  | 0.03104053 | -0.03104053 | short_liquidation_squeeze_watch | 0.12094783 | 0.241101 | funding helps direction; has recent liquidation context |
| OKX | SOL | chain_flow_reversal_watch |  | 0.11740450 | -0.11740450 | mixed_liquidation_flow_watch | 0.01447262 | 0.186926 | funding helps direction; has recent liquidation context |
| OKX | MEGA | chain_flow_reversal_watch |  | 0.28230039 | -0.28230039 |  |  | 0.182410 | funding helps direction |
| OKX | MOVE | chain_flow_reversal_watch |  | 6.09529308 | -6.09529308 |  |  | 0.155291 | funding helps direction |
| HL | SOL | chain_flow_reversal_watch |  | 0.10482829 | -0.10482829 | mixed_liquidation_flow_watch | 0.01447262 | 0.142385 | funding helps direction; has recent liquidation context |
| OKX | SUI | chain_flow_reversal_watch |  | 0.00462979 | -0.00462979 | long_liquidation_cascade_watch | 0.01842615 | 0.122863 | funding helps direction; has recent liquidation context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.02841193 | 0.109699 | has recent liquidation context |
| HL | ADA | chain_flow_reversal_watch |  | 0.27728378 | -0.27728378 | long_liquidation_cascade_watch | 0.00162827 | 0.103785 | funding helps direction |
| OKX | XLM | chain_outflow_stress_watch |  | 0.09393270 | 0.09393270 |  |  | 0.069004 | funding helps direction |
| HL | BTC | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.02636978 | 0.060147 | has recent liquidation context |
| HL | TON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.12094783 | 0.045312 | has recent liquidation context |
| OKX | BERA | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STX | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| HL | MOVE | chain_flow_reversal_watch |  | 2.87776512 | -2.87776512 |  |  | 0.000000 | funding helps direction |
| OKX | SEI | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | POL | chain_outflow_stress_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STRK | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | BNB | chain_flow_reversal_watch |  | 0.03724293 | -0.03724293 |  |  | -0.046125 | funding helps direction |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.01842615 | -0.082789 | has recent liquidation context |
| HL | AVAX | chain_flow_reversal_watch |  | 0.06176764 | -0.06176764 |  |  | -0.108977 | funding helps direction |
| OKX | AVAX | chain_flow_reversal_watch |  | -0.03845354 | 0.03845354 |  |  | -0.112358 | weak current context |
| OKX | ARB | chain_flow_reversal_watch |  | -0.05625259 | 0.05625259 |  |  | -0.116381 | weak current context |
| OKX | APT | chain_flow_reversal_watch |  | -0.05894388 | 0.05894388 |  |  | -0.132717 | weak current context |
| HL | APT | chain_flow_reversal_watch |  | 0.29968661 | -0.29968661 |  |  | -0.144955 | funding helps direction |
| OKX | MON | chain_flow_reversal_watch |  | 0.01960240 | -0.01960240 |  |  | -0.208448 | funding helps direction |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.222619 | weak current context |
| OKX | NEAR | chain_flow_reversal_watch |  | -0.06798309 | 0.06798309 | mixed_liquidation_flow_watch | 0.00484428 | -0.252261 | weak current context |
| HL | XLM | chain_outflow_stress_watch |  | 0.00314659 | 0.00314659 |  |  | -0.271327 | funding helps direction |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00484428 | -0.355113 | weak current context |
| OKX | ADA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00162827 | -0.393969 | weak current context |
| HL | ARB | chain_flow_reversal_watch |  | 0.06389982 | -0.06389982 |  |  | -0.415429 | funding helps direction |
| HL | OP | chain_flow_reversal_watch |  | 0.56179457 | -0.56179457 |  |  | -0.500000 | funding helps direction |
| HL | MEGA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| OKX | OP | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | MON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | SEI | chain_flow_reversal_watch |  | 0.15262110 | -0.15262110 |  |  | -0.847379 | funding helps direction |
