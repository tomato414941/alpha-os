# Current Protocol Fee Candidate Review

This reviews protocol fee-growth candidates against sector labels, protocol activity labels, unlock conflicts, and perp-pressure context. It is not a trade instruction.

| token | protocol | status | score | evidence | next step |
| --- | --- | --- | ---: | --- | --- |
| CRV | Curve DEX | fee_growth_unconfirmed | 63.1395 | fee_growth_7d=222.41; funding=0.0393; perp_pressure=short_carry_watch | collect another CRV fee-growth snapshot and label forward returns |
| JUP | Jupiter Perpetual Exchange | fee_growth_unconfirmed | 61.9424 | fee_growth_7d=232.55; funding=-0.1877 | collect another JUP fee-growth snapshot and label forward returns |
| MORPHO | Morpho Blue | fee_growth_unconfirmed | 49.4449 | fee_growth_7d=140.49; funding=0.1095 | collect another MORPHO fee-growth snapshot and label forward returns |
| UNI | Uniswap V3 | fee_growth_unconfirmed | 48.4203 | fee_growth_7d=126.91; funding=0.1095; perp_pressure=short_carry_watch | collect another UNI fee-growth snapshot and label forward returns |
| AAVE | Aave V3 | fee_growth_unconfirmed | 46.8366 | fee_growth_7d=129.82; funding=-0.2459; perp_pressure=long_carry_discount_watch | collect another AAVE fee-growth snapshot and label forward returns |
| ETH | Ethereum | fee_growth_unconfirmed | 41.6355 | fee_growth_7d=78.11; funding=-0.0760; perp_pressure=long_carry_discount_watch | collect another ETH fee-growth snapshot and label forward returns |
| HYPE | Hyperliquid Perps | fee_growth_unlock_conflict | 37.8965 | fee_growth_7d=111.41; funding=0.0309; perp_pressure=short_carry_premium_watch; unlock=paper_short_candidate/short | separate HYPE protocol growth thesis from unlock short pressure and label both windows |
