# Current Protocol Fee Candidate Review

This reviews protocol fee-growth candidates against sector labels, protocol activity labels, unlock conflicts, and perp-pressure context. It is not a trade instruction.

| token | protocol | status | score | evidence | next step |
| --- | --- | --- | ---: | --- | --- |
| JUP | Jupiter Perpetual Exchange | fee_growth_early_or_lagging | 55.9844 | fee_growth_7d=196.60; funding=0.0743; sector15=-0.000324 | test whether JUP fee growth leads price after short-term negative labels |
| MORPHO | Morpho Blue | fee_growth_unconfirmed | 50.6476 | fee_growth_7d=151.57; funding=0.1095 | collect another MORPHO fee-growth snapshot and label forward returns |
| AAVE | Aave V3 | fee_growth_unconfirmed | 46.7317 | fee_growth_7d=128.95; funding=-0.0844; perp_pressure=long_carry_discount_watch | collect another AAVE fee-growth snapshot and label forward returns |
| UNI | Uniswap V3 | fee_growth_unconfirmed | 43.4386 | fee_growth_7d=95.01; funding=0.1095; perp_pressure=short_carry_watch | collect another UNI fee-growth snapshot and label forward returns |
| ETH | Ethereum | fee_growth_unconfirmed | 41.5627 | fee_growth_7d=76.03; funding=-0.1570; perp_pressure=long_carry_discount_watch | collect another ETH fee-growth snapshot and label forward returns |
| HYPE | Hyperliquid Perps | fee_growth_unlock_conflict | 35.7480 | fee_growth_7d=99.26; funding=-0.0682; perp_pressure=long_carry_discount_watch; unlock=paper_short_candidate/short | separate HYPE protocol growth thesis from unlock short pressure and label both windows |
