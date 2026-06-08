# Current Lending Stress Actionability

This separates Morpho lending stress from a currently actionable lending candidate. A fully utilized market with no remaining liquidity is treated as a mechanics/risk state, not as deployable yield.

| chain | loan/collateral | status | side | score | supply USD | liquidity USD | util | avg supply APY | avg borrow APY | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ethereum | USDC/msY | lending_rate_candidate_after_risk_check | paper_lend_after_risk_check | 80.2521 | 30845405 | 3651161 | 0.8816 | 0.1119 | 0.1280 | market has visible remaining liquidity and material supply APY before protocol and withdrawal checks |
| Ethereum | USR/BONDUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 47.4180 | 7417967008 | 0 | 1.0000 | 7.5580 | 7.5580 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PAXG | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 41.1098 | 1109845736 | 0 | 1.0000 | 2978.9101 | 2978.9101 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/sdeUSD | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.6532 | 653212896 | 0 | 1.0000 | 2978.3835 | 2978.3835 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDT/USDT | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0738 | 73795283 | 0 | 1.0000 | 180.8872 | 180.8872 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/wstUSR | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0220 | 21996144 | 1 | 1.0000 | 2201.6670 | 2201.6679 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Base | USDC/HERMES | lending_stress_no_liquidity_risk | no_new_lending_until_exit_path | 40.0215 | 21480456 | 54 | 1.0000 | 2354.2596 | 2354.3058 | fully utilized market has no visible remaining liquidity, so the headline APY is not a deployable edge |
| Ethereum | USDC/PT-reUSD-25JUN2026 | lending_stress_deprioritize | none | 20.0000 | 27978779 | 2995626 | 0.8929 | 0.0624 | 0.0720 | lending state is not actionable after the basic liquidity screen |
