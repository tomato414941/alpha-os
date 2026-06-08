# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7414284832 | 7414284832 | 0 | 1.0000 | 7.5513 | 7.5513 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1113623572 | 1113623572 | 0 | 1.0000 | 2978.7015 | 2978.7015 | 102.1362 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 655436323 | 655436323 | 0 | 1.0000 | 2978.1137 | 2978.1137 | 97.5544 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 73978842 | 73978842 | 0 | 1.0000 | 180.3548 | 180.3548 | 91.7398 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 22067775 | 22067774 | 1 | 1.0000 | 2193.7498 | 2193.7506 | 91.2207 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21549100 | 21549046 | 54 | 1.0000 | 2346.7468 | 2346.7926 | 91.2154 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19405360 | 19405360 | 0 | 1.0000 | 201.3456 | 201.3456 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405308 | 19405308 | 0 | 1.0000 | 201.3462 | 201.3462 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405275 | 19405275 | 0 | 1.0000 | 201.3465 | 201.3465 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405243 | 19405243 | 0 | 1.0000 | 201.3469 | 201.3469 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405217 | 19405217 | 0 | 1.0000 | 201.3472 | 201.3472 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405184 | 19405184 | 0 | 1.0000 | 201.3475 | 201.3475 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405158 | 19405158 | 0 | 1.0000 | 201.3478 | 201.3478 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405125 | 19405125 | 0 | 1.0000 | 201.3482 | 201.3482 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405093 | 19405093 | 0 | 1.0000 | 201.3485 | 201.3485 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405060 | 19405060 | 0 | 1.0000 | 201.3489 | 201.3489 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405034 | 19405034 | 0 | 1.0000 | 201.3492 | 201.3492 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19405001 | 19405001 | 0 | 1.0000 | 201.3495 | 201.3495 | 73.1941 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19404968 | 19404968 | 0 | 1.0000 | 201.3499 | 201.3499 | 73.1940 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19404936 | 19404936 | 0 | 1.0000 | 201.3502 | 201.3502 | 73.1940 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19404910 | 19404910 | 0 | 1.0000 | 201.3505 | 201.3505 | 73.1940 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19404877 | 19404877 | 0 | 1.0000 | 201.3509 | 201.3509 | 73.1940 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19404851 | 19404851 | 0 | 1.0000 | 201.3511 | 201.3511 | 73.1940 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19404818 | 19404818 | 0 | 1.0000 | 201.3515 | 201.3515 | 73.1940 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 12987823 | 12987823 | 0 | 1.0000 | 180.3878 | 180.3878 | 73.1299 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
