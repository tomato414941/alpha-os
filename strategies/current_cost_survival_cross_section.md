# Current Cost Survival Cross Section

This ranks cost-adjusted alpha clusters by whether they survive repeat outcomes, source-lane separation, depth usage, and duplicate-pressure checks. It is a cross-sectional filter, not a live trade instruction.

| cluster | status | score | best net | mean net | lanes | split lanes | wins | losses | capacity gated | dup pressure | next probe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sui_paper_long | duplicate_pressure_control_required | 587.3191 | 258.4118 | 223.6979 | 2 | 0 | 9 | 0 | 0 | 0.5556 | dedupe SUI paper_long opportunities before any new paper ticket |
| zec_paper_long | repeat_outcome_conflicted | 490.7874 | 346.4231 | 232.3408 | 3 | 13 | 6 | 2 | 0 | 0.4444 | split ZEC paper_long by source and label winners and losers separately |
| sol_paper_long | repeat_outcome_conflicted | 477.1524 | 263.2187 | 169.4618 | 4 | 11 | 13 | 4 | 0 | 0.6500 | split SOL paper_long by source and label winners and losers separately |
| btc_paper_long | repeat_outcome_conflicted | 353.0546 | 88.6930 | 52.9868 | 3 | 22 | 6 | 1 | 0 | 0.6000 | split BTC paper_long by source and label winners and losers separately |
| chip_paper_long | cost_surviving_watchlist | 321.7968 | 346.9475 | 174.1759 | 2 | 0 | 2 | 0 | 0 | 0.5000 | paper-check CHIP paper_long with realized fill, stop, and adverse-excursion notes |
| mon_paper_long | repeat_outcome_conflicted | 195.9200 | 197.9200 | 197.9200 | 1 | 0 | 1 | 1 | 0 | 0.0000 | split MON paper_long by source and label winners and losers separately |
| link_paper_long | cost_surviving_watchlist | 195.6804 | 129.8142 | 129.8142 | 1 | 0 | 1 | 0 | 0 | 0.0000 | paper-check LINK paper_long with realized fill, stop, and adverse-excursion notes |
| fartcoin_paper_long | cost_surviving_watchlist | 181.3003 | 113.3003 | 113.3003 | 1 | 0 | 1 | 0 | 0 | 0.0000 | paper-check FARTCOIN paper_long with realized fill, stop, and adverse-excursion notes |
| eth_paper_long | repeat_outcome_conflicted | 176.3217 | 177.9382 | 105.2664 | 3 | 21 | 12 | 7 | 0 | 0.5294 | split ETH paper_long by source and label winners and losers separately |
| aave_paper_long | cost_surviving_watchlist | 165.9135 | 97.9135 | 97.9135 | 1 | 0 | 1 | 0 | 0 | 0.0000 | paper-check AAVE paper_long with realized fill, stop, and adverse-excursion notes |
| sei_paper_long | cost_surviving_watchlist | 121.1748 | 53.1748 | 53.1748 | 1 | 0 | 1 | 0 | 0 | 0.0000 | paper-check SEI paper_long with realized fill, stop, and adverse-excursion notes |
| apt_paper_long | repeat_outcome_conflicted | 66.0115 | 68.0115 | 68.0115 | 1 | 0 | 1 | 1 | 0 | 0.0000 | split APT paper_long by source and label winners and losers separately |
| hype_paper_long | repeat_outcome_conflicted | 55.6701 | 516.6457 | 182.1275 | 3 | 10 | 8 | 9 | 0 | 0.4167 | split HYPE paper_long by source and label winners and losers separately |
| bera_paper_long | capacity_blocks_cost_survival | 45.3424 | 659.1107 | 435.2402 | 2 | 0 | 2 | 0 | 2 | 0.5000 | do not scale BERA paper_long; rerun only with smaller size or deeper book |
| xpl_paper_long | capacity_blocks_cost_survival | 22.9023 | 118.7806 | 118.7806 | 1 | 0 | 1 | 0 | 1 | 0.0000 | do not scale XPL paper_long; rerun only with smaller size or deeper book |
| near_paper_long | repeat_outcome_conflicted | -315.5310 | 146.8540 | 133.4267 | 2 | 8 | 2 | 9 | 0 | 0.5000 | split NEAR paper_long by source and label winners and losers separately |
| pump_paper_long | capacity_blocks_cost_survival | -618.2254 | 47.9176 | 47.9176 | 1 | 0 | 1 | 0 | 1 | 0.0000 | do not scale PUMP paper_long; rerun only with smaller size or deeper book |

## Interpretation

High rows are not automatically good strategies. They are the rows where paper edge, repeat evidence, and cost/depth assumptions are least contradictory under the current logs.
