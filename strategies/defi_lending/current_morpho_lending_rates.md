# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7300007299 | 7300007299 | 0 | 1.0000 | 7.5614 | 7.5614 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1106670202 | 1106670202 | 0 | 1.0000 | 2979.0101 | 2979.0101 | 102.0667 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 651343936 | 651343936 | 0 | 1.0000 | 2978.5143 | 2978.5143 | 97.5134 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 73655320 | 73655320 | 0 | 1.0000 | 181.1659 | 181.1659 | 91.7366 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 21935338 | 21935337 | 1 | 1.0000 | 2205.7948 | 2205.7957 | 91.2194 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21419860 | 21419806 | 54 | 1.0000 | 2358.3045 | 2358.3509 | 91.2142 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19326800 | 19326800 | 0 | 1.0000 | 202.2361 | 202.2361 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326748 | 19326748 | 0 | 1.0000 | 202.2367 | 202.2367 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326716 | 19326716 | 0 | 1.0000 | 202.2371 | 202.2371 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326683 | 19326683 | 0 | 1.0000 | 202.2374 | 202.2374 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326657 | 19326657 | 0 | 1.0000 | 202.2377 | 202.2377 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326624 | 19326624 | 0 | 1.0000 | 202.2380 | 202.2380 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326598 | 19326598 | 0 | 1.0000 | 202.2383 | 202.2383 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326566 | 19326566 | 0 | 1.0000 | 202.2387 | 202.2387 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326533 | 19326533 | 0 | 1.0000 | 202.2390 | 202.2390 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326501 | 19326501 | 0 | 1.0000 | 202.2394 | 202.2394 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326475 | 19326475 | 0 | 1.0000 | 202.2397 | 202.2397 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326442 | 19326442 | 0 | 1.0000 | 202.2400 | 202.2400 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326410 | 19326410 | 0 | 1.0000 | 202.2404 | 202.2404 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326377 | 19326377 | 0 | 1.0000 | 202.2407 | 202.2407 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326351 | 19326351 | 0 | 1.0000 | 202.2410 | 202.2410 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326318 | 19326318 | 0 | 1.0000 | 202.2414 | 202.2414 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326292 | 19326292 | 0 | 1.0000 | 202.2417 | 202.2417 | 73.1933 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19326260 | 19326260 | 0 | 1.0000 | 202.2420 | 202.2420 | 73.1933 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 12931023 | 12931023 | 0 | 1.0000 | 181.1991 | 181.1991 | 73.1293 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
