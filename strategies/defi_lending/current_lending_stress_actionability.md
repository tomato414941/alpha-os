# Current Lending Stress Actionability

This separates Morpho lending stress from a currently actionable lending candidate. A fully utilized market with no remaining liquidity is treated as a mechanics/risk state, not as deployable yield.

| chain | loan/collateral | status | side | score | supply USD | liquidity USD | util | avg supply APY | avg borrow APY | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USDC/msY | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 80.2614 | 30826499 | 3641911 | 0.8819 | 0.1120 | 0.1279 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USDT/wstETH | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 77.8166 | 146930304 | 6175848 | 0.9580 | 0.0811 | 0.0861 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USDT/WBTC | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 76.0333 | 54856432 | 2392814 | 0.9564 | 0.0820 | 0.0870 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USR/BONDUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 47.4233 | 7423336699 | 0 | 1.0000 | 7.5455 | 7.5455 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PAXG | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 41.1176 | 1117648246 | 0 | 1.0000 | 2978.4990 | 2978.4990 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/sdeUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.6578 | 657805026 | 0 | 1.0000 | 2977.8555 | 2977.8555 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDT/USDT | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0742 | 74158076 | 0 | 1.0000 | 179.8928 | 179.8928 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/wstUSR | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0221 | 22144397 | 1 | 1.0000 | 2186.8564 | 2186.8573 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Base | USDC/HERMES | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0216 | 21624569 | 54 | 1.0000 | 2340.1212 | 2340.1666 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PT-reUSD-25JUN2026 | lending_stress_deprioritize | none | 20.0000 | 27742426 | 2908975 | 0.8951 | 0.0629 | 0.0713 | lending state is not actionable after the basic liquidity screen |
