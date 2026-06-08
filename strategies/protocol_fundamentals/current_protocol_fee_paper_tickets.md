# Current Protocol Fee Paper Tickets

This turns protocol fee-growth lag candidates that pass the current execution context gate into paper observation tickets. It is not a live trade instruction.

| token | protocol | side | score | notional | price 7d | fee growth 7d | funding | spread bps | depth 10bps USD | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AAVE | Aave V3 | long_token | 87.9820 | 1000 | -19.45 | 129.82 | 0.0664 | 4.7969 | 34036 | start AAVE paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |
| CRV | Curve DEX | long_token | 57.0025 | 1000 | -4.77 | 222.41 | 0.1095 | 3.9744 | 21208 | start CRV paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |

## Falsification

- AAVE: deprioritize AAVE if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
- CRV: deprioritize CRV if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
