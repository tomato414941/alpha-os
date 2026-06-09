# Current Lending Yield Risk Check

This checks Morpho lending yield candidates against capacity, utilization, LLTV, collateral familiarity, and APY spike risk. It is a paper-risk gate, not a deposit instruction.

| market | action | score | notional | liquidity | usage | util | LLTV | APY | avg APY | spike | collateral | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Ethereum USDT/wstETH | exit_liquidity_watch | 51.14 | 10000 | 7237899 | 0.0014 | 0.9511 | 0.86 | 0.1058 | 0.0926 | 1.14 | blue_chip_collateral | high utilization means exit and withdrawal timing dominate the headline APY |
| Ethereum USDT/WBTC | exit_liquidity_watch | 48.90 | 10000 | 2642803 | 0.0038 | 0.9520 | 0.86 | 0.1075 | 0.0929 | 1.16 | blue_chip_collateral | high utilization means exit and withdrawal timing dominate the headline APY |
| Ethereum USDC/msY | collateral_review_required | 48.37 | 10000 | 3941974 | 0.0025 | 0.8734 | 0.86 | 0.1100 | 0.1120 | 0.98 | opaque_collateral | collateral is not a familiar blue-chip, stable, or RWA symbol |
