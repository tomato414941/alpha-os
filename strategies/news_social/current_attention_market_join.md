# Current Attention Market Join

This joins CoinGecko trending attention to current Hyperliquid perp market state. It is not a trade instruction.

| symbol | name | rank | 24h change | funding | mark/oracle | carry action | obs | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| ZEC | Zcash | 2 | 21.6197 | -0.437120 | -0.001005 |  | 0 | 19.130693 | trending asset has material price move and large funding state |
| AAVE | Aave | 14 | 4.2581 | 0.109500 | -0.000707 | long_carry_reversion_watch | 6 | 18.495383 | trending asset overlaps with persistent carry/reversion perp state |

## Interpretation

Rows here combine attention and perp-market state. A row is useful only as a research candidate; it still needs future-return labels and execution checks.
