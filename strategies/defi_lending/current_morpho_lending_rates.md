# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7414549565 | 7414549565 | 0 | 1.0000 | 7.5571 | 7.5571 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1109602717 | 1109602717 | 0 | 1.0000 | 2978.8847 | 2978.8847 | 102.0960 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 653069861 | 653069861 | 0 | 1.0000 | 2978.3504 | 2978.3504 | 97.5307 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 73804623 | 73804623 | 0 | 1.0000 | 180.8179 | 180.8179 | 91.7380 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 21991175 | 21991174 | 1 | 1.0000 | 2200.6356 | 2200.6365 | 91.2199 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21473617 | 21473563 | 54 | 1.0000 | 2353.3581 | 2353.4042 | 91.2147 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19359049 | 19359049 | 0 | 1.0000 | 201.8541 | 201.8541 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358997 | 19358997 | 0 | 1.0000 | 201.8547 | 201.8547 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358964 | 19358964 | 0 | 1.0000 | 201.8551 | 201.8551 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358931 | 19358931 | 0 | 1.0000 | 201.8554 | 201.8554 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358905 | 19358905 | 0 | 1.0000 | 201.8557 | 201.8557 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358873 | 19358873 | 0 | 1.0000 | 201.8561 | 201.8561 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358847 | 19358847 | 0 | 1.0000 | 201.8563 | 201.8563 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358814 | 19358814 | 0 | 1.0000 | 201.8567 | 201.8567 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358781 | 19358781 | 0 | 1.0000 | 201.8570 | 201.8570 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358749 | 19358749 | 0 | 1.0000 | 201.8574 | 201.8574 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358723 | 19358723 | 0 | 1.0000 | 201.8577 | 201.8577 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358690 | 19358690 | 0 | 1.0000 | 201.8580 | 201.8580 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358657 | 19358657 | 0 | 1.0000 | 201.8584 | 201.8584 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358625 | 19358625 | 0 | 1.0000 | 201.8587 | 201.8587 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358599 | 19358599 | 0 | 1.0000 | 201.8590 | 201.8590 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358566 | 19358566 | 0 | 1.0000 | 201.8594 | 201.8594 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358540 | 19358540 | 0 | 1.0000 | 201.8597 | 201.8597 | 73.1936 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19358508 | 19358508 | 0 | 1.0000 | 201.8600 | 201.8600 | 73.1936 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 12957236 | 12957236 | 0 | 1.0000 | 180.8510 | 180.8510 | 73.1296 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
