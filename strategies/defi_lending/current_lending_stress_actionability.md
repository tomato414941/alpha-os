# Current Lending Stress Actionability

This separates Morpho lending stress from a currently actionable lending candidate. A fully utilized market with no remaining liquidity is treated as a mechanics/risk state, not as deployable yield.

| chain | loan/collateral | status | side | score | supply USD | liquidity USD | util | avg supply APY | avg borrow APY | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USDC/msY | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 80.4078 | 31126762 | 3941974 | 0.8734 | 0.1120 | 0.1279 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USDT/wstETH | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 79.7302 | 148008428 | 7237899 | 0.9511 | 0.0926 | 0.0977 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USDT/WBTC | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 77.4710 | 55113304 | 2642803 | 0.9520 | 0.0929 | 0.0980 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USR/BONDUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 47.4332 | 7433164798 | 0 | 1.0000 | 7.5441 | 7.5441 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PAXG | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 41.1187 | 1118664789 | 0 | 1.0000 | 2978.4452 | 2978.4452 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/sdeUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.6584 | 658403307 | 0 | 1.0000 | 2977.7875 | 2977.7875 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDT/USDT | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0742 | 74209162 | 0 | 1.0000 | 179.7775 | 179.7775 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/wstUSR | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0222 | 22163750 | 1 | 1.0000 | 2185.1319 | 2185.1328 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Base | USDC/HERMES | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0216 | 21643678 | 54 | 1.0000 | 2338.4602 | 2338.5056 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PT-reUSD-25JUN2026 | lending_stress_deprioritize | none | 20.0000 | 27615788 | 2782142 | 0.8993 | 0.0630 | 0.0713 | lending state is not actionable after the basic liquidity screen |
