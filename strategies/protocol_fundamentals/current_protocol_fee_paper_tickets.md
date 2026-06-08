# Current Protocol Fee Paper Tickets

This turns protocol fee-growth lag candidates that pass the current execution context gate into paper observation tickets. It is not a live trade instruction.

| token | protocol | side | score | notional | price 7d | fee growth 7d | funding | spread bps | depth 10bps USD | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AAVE | Aave V3 | long_token | 87.9820 | 1000 | -21.16 | 129.82 | 0.1095 | 3.7375 | 36369 | start AAVE paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |
| CRV | Curve DEX | long_token | 57.6620 | 1000 | -5.51 | 222.41 | 0.1095 | 1.9812 | 40183 | start CRV paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |

## Falsification

- AAVE: deprioritize AAVE if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
- CRV: deprioritize CRV if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
