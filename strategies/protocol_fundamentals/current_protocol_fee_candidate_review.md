# Current Protocol Fee Candidate Review

This reviews protocol fee-growth candidates against sector labels, protocol activity labels, unlock conflicts, and perp-pressure context. It is not a trade instruction.

| token | protocol | status | score | evidence | next step |
| --- | --- | --- | ---: | --- | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_unconfirmed | 57.3054 | fee_growth_7d=204.43; funding=0.0143 | collect another JUP fee-growth snapshot and label forward returns |
| MORPHO | Morpho Blue | fee_growth_unconfirmed | 48.9854 | fee_growth_7d=140.56; funding=0.0839 | collect another MORPHO fee-growth snapshot and label forward returns |
| AAVE | Aave V3 | fee_growth_unconfirmed | 46.8502 | fee_growth_7d=129.80; funding=-0.0844; perp_pressure=long_carry_discount_watch | collect another AAVE fee-growth snapshot and label forward returns |
| UNI | Uniswap V3 | fee_growth_unconfirmed | 44.8352 | fee_growth_7d=105.16; funding=0.1095; perp_pressure=short_carry_watch | collect another UNI fee-growth snapshot and label forward returns |
| HYPE | Hyperliquid Perps | fee_growth_unlock_conflict | 34.7491 | fee_growth_7d=93.34; funding=-0.0682; perp_pressure=long_carry_discount_watch; unlock=paper_short_candidate/short | separate HYPE protocol growth thesis from unlock short pressure and label both windows |
