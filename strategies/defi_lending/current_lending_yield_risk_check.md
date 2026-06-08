# Current Lending Yield Risk Check

This checks Morpho lending yield candidates against capacity, utilization, LLTV, collateral familiarity, and APY spike risk. It is a paper-risk gate, not a deposit instruction.

| market | action | score | notional | liquidity | usage | util | LLTV | APY | avg APY | spike | collateral | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Ethereum USDC/msY | collateral_review_required | 48.22 | 10000 | 3642324 | 0.0027 | 0.8819 | 0.86 | 0.1120 | 0.1120 | 1.00 | opaque_collateral | collateral is not a familiar blue-chip, stable, or RWA symbol |
| Ethereum USDT/wstETH | exit_liquidity_watch | 38.30 | 10000 | 6175968 | 0.0016 | 0.9580 | 0.86 | 0.1152 | 0.0811 | 1.42 | blue_chip_collateral | high utilization means exit and withdrawal timing dominate the headline APY |
| Ethereum USDT/WBTC | exit_liquidity_watch | 36.59 | 10000 | 2392861 | 0.0042 | 0.9564 | 0.86 | 0.1133 | 0.0820 | 1.38 | blue_chip_collateral | high utilization means exit and withdrawal timing dominate the headline APY |
