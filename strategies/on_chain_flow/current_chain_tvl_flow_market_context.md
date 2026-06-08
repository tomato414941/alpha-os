# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | ETH | chain_flow_reversal_watch |  | 0.07921879 | -0.07921879 | long_liquidation_cascade_watch | 0.05483091 | 0.576235 | funding helps direction; has recent liquidation context |
| HL | ETH | chain_flow_reversal_watch |  | -0.05222624 | 0.05222624 | long_liquidation_cascade_watch | 0.05483091 | 0.417811 | has recent liquidation context |
| OKX | BTC | chain_flow_reversal_watch |  | 0.04578734 | -0.04578734 | mixed_liquidation_flow_watch | 0.03177768 | 0.362770 | funding helps direction; has recent liquidation context |
| OKX | HYPE | chain_flow_reversal_watch |  | 0.13745376 | -0.13745376 | long_liquidation_cascade_watch | 0.02916178 | 0.348041 | funding helps direction; has recent liquidation context |
| HL | SOL | chain_flow_reversal_watch |  | 0.17846047 | -0.17846047 | mixed_liquidation_flow_watch | 0.01443639 | 0.307724 | funding helps direction; has recent liquidation context |
| OKX | TON | chain_flow_reversal_watch |  | 0.02035633 | -0.02035633 | short_liquidation_squeeze_watch | 0.12043059 | 0.232587 | funding helps direction; has recent liquidation context |
| OKX | XLM | chain_outflow_stress_watch |  | 0.09946078 | 0.09946078 | short_liquidation_squeeze_watch | 0.01530215 | 0.227630 | funding helps direction; has recent liquidation context |
| OKX | SOL | chain_flow_reversal_watch |  | 0.11794512 | -0.11794512 | mixed_liquidation_flow_watch | 0.01443639 | 0.186866 | funding helps direction; has recent liquidation context |
| HL | BTC | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.03177768 | 0.179102 | has recent liquidation context |
| OKX | MEGA | chain_flow_reversal_watch |  | 0.27208745 | -0.27208745 |  |  | 0.172476 | funding helps direction |
| OKX | MOVE | chain_flow_reversal_watch |  | 6.16096440 | -6.16096440 |  |  | 0.157417 | funding helps direction |
| OKX | SUI | chain_flow_reversal_watch |  | 0.00844594 | -0.00844594 | long_liquidation_cascade_watch | 0.02026885 | 0.144966 | funding helps direction; has recent liquidation context |
| HL | HYPE | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.02916178 | 0.094242 | has recent liquidation context |
| OKX | BERA | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STX | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | SEI | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | POL | chain_outflow_stress_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | STRK | chain_flow_reversal_watch |  |  |  |  |  | 0.000000 | weak current context |
| OKX | BNB | chain_flow_reversal_watch |  | 0.03294291 | -0.03294291 |  |  | -0.050481 | funding helps direction |
| HL | TON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.12043059 | -0.082663 | has recent liquidation context |
| OKX | AVAX | chain_flow_reversal_watch |  | -0.03820631 | 0.03820631 |  |  | -0.112319 | weak current context |
| HL | SUI | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.02026885 | -0.119430 | has recent liquidation context |
| OKX | ARB | chain_flow_reversal_watch |  | -0.06045788 | 0.06045788 |  |  | -0.120681 | weak current context |
| HL | XLM | chain_outflow_stress_watch |  | 0.10428955 | 0.10428955 | short_liquidation_squeeze_watch | 0.01530215 | -0.123400 | funding helps direction; has recent liquidation context |
| OKX | APT | chain_flow_reversal_watch |  | -0.06661740 | 0.06661740 |  |  | -0.140402 | weak current context |
| HL | ADA | chain_flow_reversal_watch |  | 0.02079274 | -0.02079274 | long_liquidation_cascade_watch | 0.00162491 | -0.171745 | funding helps direction |
| OKX | MON | chain_flow_reversal_watch |  | 0.02072809 | -0.02072809 |  |  | -0.206907 | funding helps direction |
| HL | BNB | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.243907 | weak current context |
| HL | APT | chain_flow_reversal_watch |  | 0.11547870 | -0.11547870 |  |  | -0.257211 | funding helps direction |
| OKX | NEAR | chain_flow_reversal_watch |  | -0.07990489 | 0.07990489 | mixed_liquidation_flow_watch | 0.00484394 | -0.264511 | weak current context |
| HL | AVAX | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.355868 | weak current context |
| HL | ARB | chain_flow_reversal_watch |  | 0.05839854 | -0.05839854 |  |  | -0.365638 | funding helps direction |
| OKX | ADA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00162491 | -0.394910 | weak current context |
| HL | MOVE | chain_flow_reversal_watch |  | 7.95393896 | -7.95393896 |  |  | -0.500000 | funding helps direction |
| HL | NEAR | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00484394 | -0.561061 | weak current context |
| HL | MEGA | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| OKX | OP | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | MON | chain_flow_reversal_watch |  | -0.10950000 | 0.10950000 |  |  | -0.609500 | weak current context |
| HL | OP | chain_flow_reversal_watch |  | 0.38005698 | -0.38005698 |  |  | -0.619943 | funding helps direction |
| HL | SEI | chain_flow_reversal_watch |  | 0.17257988 | -0.17257988 |  |  | -0.807516 | funding helps direction |
