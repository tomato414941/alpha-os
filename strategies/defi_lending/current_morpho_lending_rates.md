# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7423336699 | 7423336699 | 0 | 1.0000 | 7.5455 | 7.5455 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1117648246 | 1117648246 | 0 | 1.0000 | 2978.4990 | 2978.4990 | 102.1765 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 657805026 | 657805026 | 0 | 1.0000 | 2977.8555 | 2977.8555 | 97.5781 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 74158076 | 74158076 | 0 | 1.0000 | 179.8928 | 179.8928 | 91.7416 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 22144397 | 22144396 | 1 | 1.0000 | 2186.8564 | 2186.8573 | 91.2214 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21624569 | 21624515 | 54 | 1.0000 | 2340.1212 | 2340.1666 | 91.2162 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19451548 | 19451548 | 0 | 1.0000 | 200.8383 | 200.8383 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451495 | 19451495 | 0 | 1.0000 | 200.8389 | 200.8389 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451463 | 19451463 | 0 | 1.0000 | 200.8392 | 200.8392 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451430 | 19451430 | 0 | 1.0000 | 200.8396 | 200.8396 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451404 | 19451404 | 0 | 1.0000 | 200.8399 | 200.8399 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451371 | 19451371 | 0 | 1.0000 | 200.8402 | 200.8402 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451345 | 19451345 | 0 | 1.0000 | 200.8405 | 200.8405 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451312 | 19451312 | 0 | 1.0000 | 200.8408 | 200.8408 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451279 | 19451279 | 0 | 1.0000 | 200.8412 | 200.8412 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451247 | 19451247 | 0 | 1.0000 | 200.8415 | 200.8415 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451221 | 19451221 | 0 | 1.0000 | 200.8418 | 200.8418 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451188 | 19451188 | 0 | 1.0000 | 200.8422 | 200.8422 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451155 | 19451155 | 0 | 1.0000 | 200.8425 | 200.8425 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451122 | 19451122 | 0 | 1.0000 | 200.8429 | 200.8429 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451096 | 19451096 | 0 | 1.0000 | 200.8432 | 200.8432 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451063 | 19451063 | 0 | 1.0000 | 200.8435 | 200.8435 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451037 | 19451037 | 0 | 1.0000 | 200.8438 | 200.8438 | 73.1945 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19451005 | 19451005 | 0 | 1.0000 | 200.8442 | 200.8442 | 73.1945 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 13019291 | 13019291 | 0 | 1.0000 | 179.9257 | 179.9257 | 73.1302 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
