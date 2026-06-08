# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7300007299 | 7300007299 | 0 | 1.0000 | 7.5614 | 7.5614 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1106637641 | 1106637641 | 0 | 1.0000 | 2979.0101 | 2979.0101 | 102.0664 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 651324772 | 651324772 | 0 | 1.0000 | 2978.5143 | 2978.5143 | 97.5132 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 73656961 | 73656961 | 0 | 1.0000 | 181.1659 | 181.1659 | 91.7366 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 21934692 | 21934691 | 1 | 1.0000 | 2205.7948 | 2205.7957 | 91.2193 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21417166 | 21417112 | 54 | 1.0000 | 2358.3045 | 2358.3509 | 91.2141 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19324370 | 19324370 | 0 | 1.0000 | 202.2361 | 202.2361 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324318 | 19324318 | 0 | 1.0000 | 202.2367 | 202.2367 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324286 | 19324286 | 0 | 1.0000 | 202.2371 | 202.2371 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324253 | 19324253 | 0 | 1.0000 | 202.2374 | 202.2374 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324227 | 19324227 | 0 | 1.0000 | 202.2377 | 202.2377 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324194 | 19324194 | 0 | 1.0000 | 202.2380 | 202.2380 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324168 | 19324168 | 0 | 1.0000 | 202.2383 | 202.2383 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324136 | 19324136 | 0 | 1.0000 | 202.2387 | 202.2387 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324103 | 19324103 | 0 | 1.0000 | 202.2390 | 202.2390 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324071 | 19324071 | 0 | 1.0000 | 202.2394 | 202.2394 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324045 | 19324045 | 0 | 1.0000 | 202.2397 | 202.2397 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19324012 | 19324012 | 0 | 1.0000 | 202.2400 | 202.2400 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19323980 | 19323980 | 0 | 1.0000 | 202.2404 | 202.2404 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19323947 | 19323947 | 0 | 1.0000 | 202.2407 | 202.2407 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19323921 | 19323921 | 0 | 1.0000 | 202.2410 | 202.2410 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19323888 | 19323888 | 0 | 1.0000 | 202.2414 | 202.2414 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19323862 | 19323862 | 0 | 1.0000 | 202.2417 | 202.2417 | 73.1932 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19323830 | 19323830 | 0 | 1.0000 | 202.2420 | 202.2420 | 73.1932 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 12931311 | 12931311 | 0 | 1.0000 | 181.1991 | 181.1991 | 73.1293 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
