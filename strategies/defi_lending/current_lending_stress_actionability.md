# Current Lending Stress Actionability

This separates Morpho lending stress from a currently actionable lending candidate. A fully utilized market with no remaining liquidity is treated as a mechanics/risk state, not as deployable yield.

| chain | loan/collateral | status | side | score | supply USD | liquidity USD | util | avg supply APY | avg borrow APY | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USDC/msY | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 80.2390 | 30833091 | 3649695 | 0.8816 | 0.1118 | 0.1279 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USR/BONDUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 47.4145 | 7414549565 | 0 | 1.0000 | 7.5571 | 7.5571 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PAXG | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 41.1096 | 1109602717 | 0 | 1.0000 | 2978.8847 | 2978.8847 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/sdeUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.6531 | 653069861 | 0 | 1.0000 | 2978.3504 | 2978.3504 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDT/USDT | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0738 | 73804623 | 0 | 1.0000 | 180.8179 | 180.8179 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/wstUSR | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0220 | 21991175 | 1 | 1.0000 | 2200.6356 | 2200.6365 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Base | USDC/HERMES | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0215 | 21473617 | 54 | 1.0000 | 2353.3581 | 2353.4042 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PT-reUSD-25JUN2026 | lending_stress_deprioritize | none | 20.0000 | 27967561 | 2994423 | 0.8929 | 0.0654 | 0.0737 | lending state is not actionable after the basic liquidity screen |
