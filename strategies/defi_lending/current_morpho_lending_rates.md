# Current Morpho Lending Rates

This screens Morpho lending markets for borrow demand, utilization, and remaining liquidity. It is a lending-rate pressure screen, not a trade instruction.

| chain | loan | collateral | status | supply USD | borrow USD | liquidity USD | util | avg supply APY | avg borrow APY | score | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USR | BONDUSD | paper_borrow_liquidity_stress_watch | 7433164798 | 7433164798 | 0 | 1.0000 | 7.5441 | 7.5441 | 103.0000 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | PAXG | paper_borrow_liquidity_stress_watch | 1118664789 | 1118664789 | 0 | 1.0000 | 2978.4452 | 2978.4452 | 102.1866 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | sdeUSD | paper_borrow_liquidity_stress_watch | 658403307 | 658403307 | 0 | 1.0000 | 2977.7875 | 2977.7875 | 97.5840 | market is highly utilized with little remaining liquidity |
| Ethereum | USDT | USDT | paper_borrow_liquidity_stress_watch | 74209162 | 74209162 | 0 | 1.0000 | 179.7775 | 179.7775 | 91.7421 | market is highly utilized with little remaining liquidity |
| Ethereum | USDC | wstUSR | paper_borrow_liquidity_stress_watch | 22163750 | 22163749 | 1 | 1.0000 | 2185.1319 | 2185.1328 | 91.2216 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | paper_borrow_liquidity_stress_watch | 21643678 | 21643624 | 54 | 1.0000 | 2338.4602 | 2338.5056 | 91.2164 | market is highly utilized with little remaining liquidity |
| Base | USDC | HERMES | lending_context_watch | 19463279 | 19463279 | 0 | 1.0000 | 200.7115 | 200.7115 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19463227 | 19463227 | 0 | 1.0000 | 200.7120 | 200.7120 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19463194 | 19463194 | 0 | 1.0000 | 200.7124 | 200.7124 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19463161 | 19463161 | 0 | 1.0000 | 200.7128 | 200.7128 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19463135 | 19463135 | 0 | 1.0000 | 200.7130 | 200.7130 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19463102 | 19463102 | 0 | 1.0000 | 200.7134 | 200.7134 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19463076 | 19463076 | 0 | 1.0000 | 200.7137 | 200.7137 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19463043 | 19463043 | 0 | 1.0000 | 200.7140 | 200.7140 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19463010 | 19463010 | 0 | 1.0000 | 200.7144 | 200.7144 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462978 | 19462978 | 0 | 1.0000 | 200.7147 | 200.7147 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462952 | 19462952 | 0 | 1.0000 | 200.7150 | 200.7150 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462919 | 19462919 | 0 | 1.0000 | 200.7154 | 200.7154 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462886 | 19462886 | 0 | 1.0000 | 200.7157 | 200.7157 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462853 | 19462853 | 0 | 1.0000 | 200.7161 | 200.7161 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462827 | 19462827 | 0 | 1.0000 | 200.7163 | 200.7163 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462794 | 19462794 | 0 | 1.0000 | 200.7167 | 200.7167 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462768 | 19462768 | 0 | 1.0000 | 200.7170 | 200.7170 | 73.1946 | lending market context exists but is not yet actionable |
| Base | USDC | HERMES | lending_context_watch | 19462735 | 19462735 | 0 | 1.0000 | 200.7173 | 200.7173 | 73.1946 | lending market context exists but is not yet actionable |
| Ethereum | USDT | USDT | lending_context_watch | 13028260 | 13028260 | 0 | 1.0000 | 179.8104 | 179.8104 | 73.1303 | lending market context exists but is not yet actionable |

## Interpretation

High utilization and high borrow APY can indicate leverage demand or liquidity stress. A lending candidate still needs rate persistence, collateral drawdown, oracle, liquidation, withdrawal, gas, and smart-contract risk checks.
