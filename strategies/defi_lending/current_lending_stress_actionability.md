# Current Lending Stress Actionability

This separates Morpho lending stress from a currently actionable lending candidate. A fully utilized market with no remaining liquidity is treated as a mechanics/risk state, not as deployable yield.

| chain | loan/collateral | status | side | score | supply USD | liquidity USD | util | avg supply APY | avg borrow APY | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USDC/msY | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 80.3405 | 30879836 | 3695025 | 0.8803 | 0.1124 | 0.1283 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USR/BONDUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 47.3000 | 7300007299 | 0 | 1.0000 | 7.5614 | 7.5614 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PAXG | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 41.1067 | 1106670202 | 0 | 1.0000 | 2979.0101 | 2979.0101 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/sdeUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.6513 | 651343936 | 0 | 1.0000 | 2978.5143 | 2978.5143 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDT/USDT | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0737 | 73655320 | 0 | 1.0000 | 181.1659 | 181.1659 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/wstUSR | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0219 | 21935338 | 1 | 1.0000 | 2205.7948 | 2205.7957 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Base | USDC/HERMES | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0214 | 21419860 | 54 | 1.0000 | 2358.3045 | 2358.3509 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PT-reUSD-25JUN2026 | lending_stress_deprioritize | none | 20.0000 | 29013498 | 4040591 | 0.8607 | 0.0612 | 0.0717 | lending state is not actionable after the basic liquidity screen |
