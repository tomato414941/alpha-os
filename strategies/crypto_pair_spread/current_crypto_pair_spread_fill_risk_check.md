# Current Crypto Pair Spread Fill Risk Check

This checks pair-ratio mark wins against rough two-leg spread, taker fee, funding, and visible-depth assumptions. It is not a fill report.

| ticket | pair | dir bps | cost bps | funding 1h | net bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| pair-spread-eth-hype-mean-reversion | ETH/HYPE | 17.50423490 | 16.74809508 | -0.23893800 | 0.51720182 | pair_cost_adjusted_edge_failed | pair mark win does not survive conservative two-leg taker cost |
| pair-spread-eth-sol-mean-reversion | ETH/SOL | 11.93882652 | 16.73931901 | -0.05387800 | -4.85437049 | pair_cost_adjusted_edge_failed | pair mark win does not survive conservative two-leg taker cost |
| pair-spread-sol-hype-mean-reversion | SOL/HYPE | 5.55877186 | 16.30764709 | -0.18506000 | -10.93393523 | pair_cost_adjusted_edge_failed | pair mark win does not survive conservative two-leg taker cost |
