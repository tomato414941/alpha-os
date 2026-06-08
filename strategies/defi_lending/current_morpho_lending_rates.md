# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7423336699 | 7423336699 | 0 | 1.0000 | 7.5455 | 7.5455 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1117774916 | 1117774916 | 0 | 1.0000 | 2978.4990 | 2978.4990 | 102.1777 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 657879579 | 657879579 | 0 | 1.0000 | 2977.8555 | 2977.8555 | 97.5788 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 74159517 | 74159517 | 0 | 1.0000 | 179.8928 | 179.8928 | 91.7416 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 22146907 | 22146906 | 1 | 1.0000 | 2186.8564 | 2186.8573 | 91.2215 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21627020 | 21626966 | 54 | 1.0000 | 2340.1212 | 2340.1666 | 91.2162 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19453752 | 19453752 | 0 | 1.0000 | 200.8383 | 200.8383 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453700 | 19453700 | 0 | 1.0000 | 200.8389 | 200.8389 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453667 | 19453667 | 0 | 1.0000 | 200.8392 | 200.8392 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453635 | 19453635 | 0 | 1.0000 | 200.8396 | 200.8396 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453608 | 19453608 | 0 | 1.0000 | 200.8399 | 200.8399 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453576 | 19453576 | 0 | 1.0000 | 200.8402 | 200.8402 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453549 | 19453549 | 0 | 1.0000 | 200.8405 | 200.8405 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453517 | 19453517 | 0 | 1.0000 | 200.8408 | 200.8408 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453484 | 19453484 | 0 | 1.0000 | 200.8412 | 200.8412 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453451 | 19453451 | 0 | 1.0000 | 200.8415 | 200.8415 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453425 | 19453425 | 0 | 1.0000 | 200.8418 | 200.8418 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453392 | 19453392 | 0 | 1.0000 | 200.8422 | 200.8422 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453360 | 19453360 | 0 | 1.0000 | 200.8425 | 200.8425 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453327 | 19453327 | 0 | 1.0000 | 200.8429 | 200.8429 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453301 | 19453301 | 0 | 1.0000 | 200.8432 | 200.8432 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453268 | 19453268 | 0 | 1.0000 | 200.8435 | 200.8435 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453242 | 19453242 | 0 | 1.0000 | 200.8438 | 200.8438 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19453209 | 19453209 | 0 | 1.0000 | 200.8442 | 200.8442 | 73.1945 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 13019544 | 13019544 | 0 | 1.0000 | 179.9257 | 179.9257 | 73.1302 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
