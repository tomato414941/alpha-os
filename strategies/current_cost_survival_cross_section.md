# Current Cost Survival Cross Section

This ranks cost-adjusted alpha clusters by whether they survive repeat outcomes, source-lane separation, depth usage, and duplicate-pressure checks. It is a cross-sectional filter, not a live trade instruction.

| cluster | status | score | best net | mean net | lanes | split lanes | wins | losses | capacity gated | dup pressure | next probe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| zec_paper_long | cost_surviving_cross_section_leader | 806.6875 | 703.2161 | 415.6534 | 5 | 13 | 11 | 0 | 0 | 0.5312 | paper-check ZEC paper_long with realized fill, stop, and adverse-excursion notes |
| near_paper_long | cost_surviving_cross_section_leader | 662.3251 | 198.0359 | 73.4281 | 3 | 8 | 11 | 0 | 0 | 0.2353 | paper-check NEAR paper_long with realized fill, stop, and adverse-excursion notes |
| chip_paper_long | cost_surviving_cross_section_leader | 454.5000 | 655.8506 | 327.0326 | 4 | 0 | 2 | 0 | 0 | 0.1111 | paper-check CHIP paper_long with realized fill, stop, and adverse-excursion notes |
| bera_paper_long | cost_surviving_watchlist | 363.5000 | 743.4756 | 517.9148 | 2 | 0 | 2 | 0 | 0 | 0.5000 | paper-check BERA paper_long with realized fill, stop, and adverse-excursion notes |
| sui_paper_long | repeat_outcome_conflicted | 311.9428 | 132.1735 | 111.7541 | 2 | 0 | 8 | 2 | 0 | 0.5000 | split SUI paper_long by source and label winners and losers separately |
| pol_paper_short | cost_adjusted_but_unrepeated | 292.0100 | 281.1334 | 281.1334 | 1 | 0 | 0 | 0 | 0 | 0.0000 | open one repeat probe for POL paper_short before ranking it against leaders |
| pump_paper_long | cost_surviving_watchlist | 289.9657 | 221.9657 | 221.9657 | 1 | 0 | 1 | 0 | 0 | 0.0000 | paper-check PUMP paper_long with realized fill, stop, and adverse-excursion notes |
| inj_paper_long | cost_surviving_watchlist | 259.5729 | 63.8840 | 63.8840 | 1 | 0 | 5 | 0 | 0 | 0.2000 | paper-check INJ paper_long with realized fill, stop, and adverse-excursion notes |
| apt_paper_long | duplicate_pressure_control_required | 198.9535 | 94.3342 | 36.3693 | 3 | 0 | 3 | 0 | 0 | 0.7500 | dedupe APT paper_long opportunities before any new paper ticket |
| sol_paper_long | repeat_outcome_conflicted | 193.3102 | 259.3567 | 165.6359 | 4 | 11 | 13 | 8 | 0 | 0.6500 | split SOL paper_long by source and label winners and losers separately |
| xpl_paper_long | cost_surviving_watchlist | 188.4125 | 120.4125 | 120.4125 | 1 | 0 | 1 | 0 | 0 | 0.0000 | paper-check XPL paper_long with realized fill, stop, and adverse-excursion notes |
| btc_paper_short | cost_adjusted_but_unrepeated | 104.9184 | 76.9184 | 76.9184 | 1 | 0 | 0 | 0 | 0 | 0.0000 | open one repeat probe for BTC paper_short before ranking it against leaders |
| link_paper_long | repeat_outcome_conflicted | 97.1776 | 99.1776 | 99.1776 | 1 | 0 | 1 | 1 | 0 | 0.0000 | split LINK paper_long by source and label winners and losers separately |
| eth_paper_short | cost_adjusted_but_unrepeated | 77.5783 | 49.5783 | 49.5783 | 1 | 0 | 0 | 0 | 0 | 0.0000 | open one repeat probe for ETH paper_short before ranking it against leaders |
| uni_paper_long | cost_adjusted_but_unrepeated | 73.9608 | 75.9608 | 75.9608 | 1 | 0 | 0 | 0 | 0 | 0.3333 | open one repeat probe for UNI paper_long before ranking it against leaders |
| doge_paper_short | cost_adjusted_but_unrepeated | 36.9292 | 8.9292 | 8.9292 | 1 | 0 | 0 | 0 | 0 | 0.0000 | open one repeat probe for DOGE paper_short before ranking it against leaders |
| fartcoin_paper_long | capacity_blocks_cost_survival | -108.7427 | 53.8058 | 53.8058 | 1 | 0 | 1 | 1 | 1 | 0.0000 | do not scale FARTCOIN paper_long; rerun only with smaller size or deeper book |
| eth_paper_long | repeat_outcome_conflicted | -443.2294 | 118.8000 | 49.6556 | 3 | 21 | 12 | 15 | 0 | 0.5556 | split ETH paper_long by source and label winners and losers separately |
| btc_paper_long | repeat_outcome_conflicted | -515.0831 | 2.9169 | 2.9169 | 1 | 22 | 0 | 9 | 0 | 0.0000 | split BTC paper_long by source and label winners and losers separately |
| hype_paper_long | repeat_outcome_conflicted | -1085.5127 | 299.9717 | 299.9717 | 1 | 10 | 0 | 21 | 0 | 0.0000 | split HYPE paper_long by source and label winners and losers separately |

## Interpretation

High rows are not automatically good strategies. They are the rows where paper edge, repeat evidence, and cost/depth assumptions are least contradictory under the current logs.
