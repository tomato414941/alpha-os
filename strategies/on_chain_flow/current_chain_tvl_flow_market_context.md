# Current Chain TVL Flow Market Context

This joins chain TVL flow forward labels with current perp funding, liquidity, and OKX liquidation context. It is still a research screen, not a deployable strategy.

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | SOL | chain_flow_reversal_watch | 0.00107280 | 0.38634890 | -0.38634890 |  |  | 0.417124 | price label positive; funding helps direction |
| OKX | HYPE | chain_flow_reversal_watch | 0.00406987 | 0.06081031 | -0.06081031 | mixed_liquidation_flow_watch | 0.00082611 | 0.391720 | price label positive; funding helps direction |
| OKX | ETH | chain_flow_reversal_watch | 0.00032510 | 0.15622280 | -0.15622280 | short_liquidation_squeeze_watch | 0.01962368 | 0.381906 | price label positive; funding helps direction; has recent liquidation context |
| OKX | MEGA | chain_flow_reversal_watch | 0.00206697 | 0.26241290 | -0.26241290 |  |  | 0.366388 | price label positive; funding helps direction |
| HL | HYPE | chain_flow_reversal_watch | 0.00428494 | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00082611 | 0.318829 | price label positive |
| HL | ETH | chain_flow_reversal_watch | 0.00030667 | 0.13897127 | -0.13897127 | short_liquidation_squeeze_watch | 0.01962368 | 0.313806 | price label positive; funding helps direction; has recent liquidation context |
| OKX | KAT | chain_flow_reversal_watch | 0.00310258 |  |  |  |  | 0.310258 | price label positive |
| OKX | POL | chain_flow_reversal_watch | 0.00253036 |  |  |  |  | 0.253036 | price label positive |
| HL | SOL | chain_flow_reversal_watch | 0.00085786 | 0.15540240 | -0.15540240 |  |  | 0.233541 | price label positive; funding helps direction |
| OKX | STX | chain_flow_reversal_watch | 0.00107875 |  |  |  |  | 0.107875 | price label positive |
| OKX | STRK | chain_outflow_stress_watch | 0.00060078 |  |  |  |  | 0.060078 | price label positive |
| OKX | BTC | chain_flow_reversal_watch | 0.00049420 | -0.00963477 | 0.00963477 | short_liquidation_squeeze_watch | 0.00059717 | 0.044953 | price label positive |
| OKX | MON | chain_flow_reversal_watch | -0.00229043 | 0.80572183 | -0.80572183 |  |  | 0.042281 | funding helps direction |
| OKX | MOVE | chain_flow_reversal_watch | 0.00000000 |  |  |  |  | 0.000000 | weak current context |
| OKX | APT | chain_flow_reversal_watch | 0.00089713 | -0.01920559 | 0.01920559 |  |  | -0.004058 | price label positive |
| OKX | SEI | chain_flow_reversal_watch | -0.00020338 |  |  |  |  | -0.020338 | weak current context |
| OKX | BERA | chain_flow_reversal_watch | -0.00041203 |  |  |  |  | -0.041203 | weak current context |
| HL | BTC | chain_flow_reversal_watch | 0.00053108 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00059717 | -0.058458 | price label positive |
| OKX | BNB | chain_flow_reversal_watch | 0.00100705 | -0.07965044 | 0.07965044 |  |  | -0.062719 | price label positive |
| OKX | ADA | chain_flow_reversal_watch | 0.00061425 | 0.16470644 | -0.16470644 | short_liquidation_squeeze_watch | 0.00044604 | -0.075687 | price label positive; funding helps direction |
| OKX | TON | chain_flow_reversal_watch | -0.00173110 | -0.03393936 | 0.03393936 | short_liquidation_squeeze_watch | 0.04191457 | -0.076504 | has recent liquidation context |
| OKX | ARB | chain_flow_reversal_watch | -0.00048894 | 0.00013278 | -0.00013278 |  |  | -0.109785 | funding helps direction |
| HL | BNB | chain_flow_reversal_watch | 0.00070475 | -0.10950000 | 0.10950000 |  |  | -0.144539 | price label positive |
| HL | ADA | chain_flow_reversal_watch | 0.00159637 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00044604 | -0.156580 | price label positive |
| HL | TON | chain_flow_reversal_watch | -0.00069144 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.04191457 | -0.189005 | has recent liquidation context |
| HL | MEGA | chain_flow_reversal_watch | 0.00248144 | 0.01124784 | -0.01124784 |  |  | -0.240608 | price label positive; funding helps direction |
| OKX | SUI | chain_flow_reversal_watch | -0.00133404 | -0.06427009 | 0.06427009 | short_liquidation_squeeze_watch | 0.00155074 | -0.248873 | weak current context |
| OKX | OP | chain_flow_reversal_watch | 0.00210305 | 0.01300102 | -0.01300102 |  |  | -0.276694 | price label positive; funding helps direction |
| HL | AVAX | chain_outflow_stress_watch | -0.00190049 | 0.06122101 | 0.06122101 |  |  | -0.300334 | funding helps direction |
| HL | APT | chain_flow_reversal_watch | 0.00134549 | -0.10950000 | 0.10950000 |  |  | -0.347640 | price label positive |
| OKX | AVAX | chain_outflow_stress_watch | -0.00194582 | -0.09662321 | -0.09662321 |  |  | -0.365782 | weak current context |
| HL | SUI | chain_flow_reversal_watch | -0.00163928 | -0.10950000 | 0.10950000 | short_liquidation_squeeze_watch | 0.00155074 | -0.371225 | weak current context |
| HL | SEI | chain_flow_reversal_watch | 0.00089509 | 0.74291020 | -0.74291020 |  |  | -0.377250 | price label positive; funding helps direction |
| OKX | NEAR | chain_outflow_stress_watch | -0.00392927 | -0.03291537 | -0.03291537 | long_liquidation_cascade_watch | 0.00985181 | -0.571525 | weak current context |
| HL | ARB | chain_flow_reversal_watch | 0.00036679 | -0.10950000 | 0.10950000 |  |  | -0.572821 | price label positive |
| HL | NEAR | chain_outflow_stress_watch | -0.00441956 | 0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00985181 | -0.631741 | funding helps direction |
| OKX | XLM | chain_outflow_stress_watch | -0.00340136 | -0.13856514 | -0.13856514 | long_liquidation_cascade_watch | 0.00031398 | -0.717399 | weak current context |
| HL | XLM | chain_outflow_stress_watch | -0.00364325 | 0.10950000 | 0.10950000 | long_liquidation_cascade_watch | 0.00031398 | -0.725573 | funding helps direction |
| HL | MON | chain_flow_reversal_watch | -0.00228331 | -0.10950000 | 0.10950000 |  |  | -0.748471 | weak current context |
| HL | POL | chain_flow_reversal_watch | 0.00140398 | -0.01380488 | 0.01380488 |  |  | -0.770888 | price label positive |
