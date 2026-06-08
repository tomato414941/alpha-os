# Current Protocol Fee Paper Tickets

This turns protocol fee-growth lag candidates that pass the current execution context gate into paper observation tickets. It is not a live trade instruction.

| token | protocol | side | score | notional | price 7d | fee growth 7d | funding | spread bps | depth 10bps USD | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AAVE | Aave V3 | long_token | 87.9820 | 1000 | -20.09 | 129.82 | 0.0197 | 2.4959 | 35915 | start AAVE paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |
| CRV | Curve DEX | long_token | 59.0582 | 1000 | -6.71 | 222.41 | 0.1095 | 0.5009 | 23800 | start CRV paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |
| MORPHO | Morpho Blue | long_token | 54.8658 | 1000 | -12.57 | 140.49 | 0.1095 | 3.8232 | 6508 | start MORPHO paper observation now and label 4h,12h,24h returns with funding, spread, and depth context |

## Falsification

- AAVE: deprioritize AAVE if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
- CRV: deprioritize CRV if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
- MORPHO: deprioritize MORPHO if 4h and 12h directional labels fail, or if fresh venue context becomes thin, wide, or unavailable.
