# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7448150258 | 7448150258 | 0 | 1.0000 | 7.6058 | 7.6058 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1091579231 | 1091579231 | 0 | 1.0000 | 2979.7496 | 2979.7496 | 101.9158 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 642462128 | 642462128 | 0 | 1.0000 | 2979.5596 | 2979.5596 | 97.4246 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 73021664 | 73021664 | 0 | 1.0000 | 184.7384 | 184.7384 | 91.7302 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 21647225 | 21647224 | 1 | 1.0000 | 2257.8490 | 2257.8499 | 91.2165 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21134079 | 21134025 | 54 | 1.0000 | 2407.8975 | 2407.9463 | 91.2113 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19149182 | 19149182 | 0 | 1.0000 | 206.1526 | 206.1526 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19149130 | 19149130 | 0 | 1.0000 | 206.1531 | 206.1531 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19149098 | 19149098 | 0 | 1.0000 | 206.1535 | 206.1535 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19149066 | 19149066 | 0 | 1.0000 | 206.1538 | 206.1538 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19149040 | 19149040 | 0 | 1.0000 | 206.1541 | 206.1541 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19149007 | 19149007 | 0 | 1.0000 | 206.1545 | 206.1545 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148982 | 19148982 | 0 | 1.0000 | 206.1548 | 206.1548 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148949 | 19148949 | 0 | 1.0000 | 206.1551 | 206.1551 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148917 | 19148917 | 0 | 1.0000 | 206.1555 | 206.1555 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148885 | 19148885 | 0 | 1.0000 | 206.1559 | 206.1559 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148859 | 19148859 | 0 | 1.0000 | 206.1562 | 206.1562 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148826 | 19148826 | 0 | 1.0000 | 206.1565 | 206.1565 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148794 | 19148794 | 0 | 1.0000 | 206.1569 | 206.1569 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148762 | 19148762 | 0 | 1.0000 | 206.1572 | 206.1572 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148736 | 19148736 | 0 | 1.0000 | 206.1575 | 206.1575 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148704 | 19148704 | 0 | 1.0000 | 206.1579 | 206.1579 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148678 | 19148678 | 0 | 1.0000 | 206.1582 | 206.1582 | 73.1915 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19148645 | 19148645 | 0 | 1.0000 | 206.1585 | 206.1585 | 73.1915 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 12819774 | 12819774 | 0 | 1.0000 | 184.7726 | 184.7726 | 73.1282 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
