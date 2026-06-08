# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7417967008 | 7417967008 | 0 | 1.0000 | 7.5580 | 7.5580 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1109845736 | 1109845736 | 0 | 1.0000 | 2978.9101 | 2978.9101 | 102.0985 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 653212896 | 653212896 | 0 | 1.0000 | 2978.3835 | 2978.3835 | 97.5321 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 73795283 | 73795283 | 0 | 1.0000 | 180.8872 | 180.8872 | 91.7380 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 21996144 | 21996143 | 1 | 1.0000 | 2201.6670 | 2201.6679 | 91.2200 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21480456 | 21480402 | 54 | 1.0000 | 2354.2596 | 2354.3058 | 91.2148 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19365723 | 19365723 | 0 | 1.0000 | 201.9237 | 201.9237 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365671 | 19365671 | 0 | 1.0000 | 201.9243 | 201.9243 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365638 | 19365638 | 0 | 1.0000 | 201.9247 | 201.9247 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365606 | 19365606 | 0 | 1.0000 | 201.9250 | 201.9250 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365580 | 19365580 | 0 | 1.0000 | 201.9253 | 201.9253 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365547 | 19365547 | 0 | 1.0000 | 201.9257 | 201.9257 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365521 | 19365521 | 0 | 1.0000 | 201.9259 | 201.9259 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365488 | 19365488 | 0 | 1.0000 | 201.9263 | 201.9263 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365456 | 19365456 | 0 | 1.0000 | 201.9266 | 201.9266 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365423 | 19365423 | 0 | 1.0000 | 201.9270 | 201.9270 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365397 | 19365397 | 0 | 1.0000 | 201.9273 | 201.9273 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365364 | 19365364 | 0 | 1.0000 | 201.9276 | 201.9276 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365332 | 19365332 | 0 | 1.0000 | 201.9280 | 201.9280 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365299 | 19365299 | 0 | 1.0000 | 201.9283 | 201.9283 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365273 | 19365273 | 0 | 1.0000 | 201.9286 | 201.9286 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365240 | 19365240 | 0 | 1.0000 | 201.9290 | 201.9290 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365214 | 19365214 | 0 | 1.0000 | 201.9293 | 201.9293 | 73.1937 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19365182 | 19365182 | 0 | 1.0000 | 201.9296 | 201.9296 | 73.1937 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 12955596 | 12955596 | 0 | 1.0000 | 180.9204 | 180.9204 | 73.1296 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
