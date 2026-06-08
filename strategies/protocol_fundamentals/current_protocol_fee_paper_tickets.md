# Current Protocol Fee Paper Tickets

This turns protocol fee-growth lag candidates that pass the current execution context gate into paper observation tickets. It is not a live trade instruction.

| token | protocol | side | score | notional | price 7d | fee growth 7d | funding | spread bps | depth 10bps USD | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AAVE | Aave V3 | long_token | 87.9800 | 1000 | -21.82 | 129.80 | -0.0505 | 2.6867 | 42507 | start AAVE paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |
| CRV | Curve DEX | long_token | 59.9474 | 1000 | -8.57 | 210.10 | 0.1095 | 1.5021 | 20772 | start CRV paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |

## Falsification

- AAVE: deprioritize AAVE if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
- CRV: deprioritize CRV if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
