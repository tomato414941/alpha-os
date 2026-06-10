# Current Crypto Pair Spread Fill Risk Check

This checks pair-ratio mark wins against rough two-leg spread, taker fee, funding, and visible-depth assumptions. It is not a fill report.

| ticket | pair | dir bps | cost bps | funding 1h | net bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| pair-spread-btc-sol-mean-reversion | BTC/SOL | 103.06080227 | 16.31277928 | 0.30602000 | 87.05404299 | cost_adjusted_pair_probe | pair mark win survives conservative two-leg taker cost |
| pair-spread-eth-sol-mean-reversion | ETH/SOL | 63.03134898 | 17.35493629 | 0.12922600 | 45.80563869 | cost_adjusted_pair_probe | pair mark win survives conservative two-leg taker cost |
| pair-spread-btc-eth-mean-reversion | BTC/ETH | 39.77872264 | 17.36226713 | 0.17679400 | 22.59324951 | cost_adjusted_pair_probe | pair mark win survives conservative two-leg taker cost |
| pair-spread-btc-hype-mean-reversion | BTC/HYPE | 22.56721289 | 16.48104113 | -0.04957500 | 6.03659676 | cost_adjusted_pair_probe | pair mark win survives conservative two-leg taker cost |
