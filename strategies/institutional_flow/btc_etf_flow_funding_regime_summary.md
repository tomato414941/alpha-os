# BTC ETF Flow Funding Regime Summary

This joins leakage-safe BTC ETF flow labels to Binance BTCUSDT perp funding. Positive funding means BTC perp shorts receive funding; negative funding means longs receive funding. Funding PnL is a rough daily notional proxy, not an execution-ready PnL.

| group | obs | mean 5d flow BTC | start funding | 5d funding | funding support | dir 5d | dir 5d + funding | hit 5d + funding | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| large_5d_outflow__funding_aligned | 45 | -24371.84 | 0.00013556 | 0.00057130 | 0.00057130 | 0.03107124 | 0.03164254 | 0.6889 | funding_regime_candidate |
| large_5d_outflow | 51 | -25190.65 | 0.00010456 | 0.00044689 | 0.00044689 | 0.03057730 | 0.03102419 | 0.6863 | funding_regime_candidate |
| large_5d_outflow__funding_against | 6 | -31331.67 | -0.00012789 | -0.00048615 | -0.00048615 | 0.02687275 | 0.02638660 | 0.6667 | weak_or_insufficient |
| btc_etf_distribution_label__funding_aligned | 76 | -16139.32 | 0.00015133 | 0.00070996 | 0.00070996 | 0.01685245 | 0.01756240 | 0.5921 | funding_regime_watch |
| btc_etf_distribution_label | 86 | -16608.08 | 0.00012354 | 0.00058594 | 0.00058594 | 0.01513873 | 0.01572468 | 0.5814 | funding_regime_watch |
| btc_etf_inflow_context_label__funding_against | 108 | 17147.93 | 0.00025780 | 0.00124219 | -0.00124219 | 0.01684801 | 0.01560582 | 0.5741 | weak_or_insufficient |
| btc_etf_inflow_context_label | 139 | 15883.07 | 0.00020652 | 0.00098101 | -0.00087194 | 0.01637483 | 0.01550289 | 0.6115 | weak_or_insufficient |
| btc_etf_inflow_context_label__funding_aligned | 31 | 11476.48 | 0.00002786 | 0.00007111 | 0.00041794 | 0.01472634 | 0.01514428 | 0.7419 | funding_regime_candidate |
| large_5d_inflow__funding_against | 196 | 36844.45 | 0.00033699 | 0.00165400 | -0.00165400 | 0.01637724 | 0.01472324 | 0.5306 | weak_or_insufficient |
| large_5d_inflow | 226 | 36532.69 | 0.00028221 | 0.00138741 | -0.00138741 | 0.01522096 | 0.01383354 | 0.5531 | weak_or_insufficient |
| btc_etf_outflow_context_label__funding_aligned | 64 | -9394.33 | 0.00014749 | 0.00071158 | 0.00071203 | 0.01090077 | 0.01161280 | 0.5625 | funding_regime_watch |
| btc_etf_accumulation_label__funding_against | 209 | 30334.51 | 0.00029060 | 0.00148028 | -0.00148028 | 0.01160635 | 0.01012608 | 0.5455 | weak_or_insufficient |
| btc_etf_accumulation_label | 241 | 29708.39 | 0.00024150 | 0.00123905 | -0.00123905 | 0.01089986 | 0.00966081 | 0.5602 | weak_or_insufficient |
| mixed_5d_flow__funding_aligned | 128 | -4612.40 | 0.00012062 | 0.00059263 | 0.00067900 | 0.00787414 | 0.00855314 | 0.5703 | funding_regime_watch |
| large_5d_inflow__funding_aligned | 30 | 34495.87 | -0.00007567 | -0.00035428 | 0.00035428 | 0.00766654 | 0.00802083 | 0.7000 | funding_regime_watch |
| btc_etf_accumulation_label__funding_aligned | 32 | 25619.06 | -0.00007919 | -0.00033643 | 0.00033643 | 0.00628562 | 0.00662205 | 0.6562 | funding_regime_watch |
| btc_etf_outflow_context_label | 86 | -8585.24 | 0.00012644 | 0.00062051 | 0.00036360 | 0.00560175 | 0.00596535 | 0.4535 | funding_regime_watch |
| mixed_5d_flow | 275 | 833.37 | 0.00014288 | 0.00073593 | -0.00017232 | 0.00613554 | 0.00596322 | 0.5418 | weak_or_insufficient |
| mixed_5d_flow__funding_against | 147 | 5575.27 | 0.00016226 | 0.00086071 | -0.00091360 | 0.00462165 | 0.00370805 | 0.5170 | weak_or_insufficient |
| btc_etf_distribution_label__funding_against | 10 | -20170.70 | -0.00008770 | -0.00035656 | -0.00035656 | 0.00211450 | 0.00175794 | 0.5000 | weak_or_insufficient |
| btc_etf_outflow_context_label__funding_against | 22 | -6231.55 | 0.00006520 | 0.00035558 | -0.00065003 | -0.00981356 | -0.01046359 | 0.1364 | weak_or_insufficient |

## Interpretation

The main tradable question is whether ETF flow direction and perp funding carry point the same way. For example, large ETF outflow plus positive BTCUSDT funding means the short BTC view also receives funding.
