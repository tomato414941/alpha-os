# BTC ETF Flow Funding Regime Summary

This joins leakage-safe BTC ETF flow labels to Binance BTCUSDT perp funding. Positive funding means BTC perp shorts receive funding; negative funding means longs receive funding. Regime alignment is based on the label-start funding rate, while 5-day funding is only used as rough PnL.

| group | obs | mean 5d flow BTC | start funding | 5d funding | funding support | dir 5d | dir 5d + funding | hit 5d + funding | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| large_5d_outflow__funding_aligned | 43 | -23844.07 | 0.00014644 | 0.00056497 | 0.00056497 | 0.03206919 | 0.03263416 | 0.6977 | funding_regime_candidate |
| large_5d_outflow | 51 | -25190.65 | 0.00010456 | 0.00044689 | 0.00044689 | 0.03057730 | 0.03102419 | 0.6863 | funding_regime_candidate |
| large_5d_outflow__funding_against | 8 | -32428.50 | -0.00012051 | -0.00018778 | -0.00018778 | 0.02255838 | 0.02237060 | 0.6250 | weak_or_insufficient |
| btc_etf_distribution_label__funding_aligned | 73 | -15533.64 | 0.00016177 | 0.00071181 | 0.00071181 | 0.01733388 | 0.01804569 | 0.6027 | funding_regime_candidate |
| btc_etf_inflow_context_label__funding_against | 110 | 16469.51 | 0.00025879 | 0.00118226 | -0.00118226 | 0.01747954 | 0.01629728 | 0.5909 | weak_or_insufficient |
| btc_etf_distribution_label | 86 | -16608.08 | 0.00012354 | 0.00058594 | 0.00058594 | 0.01513873 | 0.01572468 | 0.5814 | funding_regime_watch |
| btc_etf_inflow_context_label | 139 | 15883.07 | 0.00020652 | 0.00098101 | -0.00087194 | 0.01637483 | 0.01550289 | 0.6115 | weak_or_insufficient |
| large_5d_inflow__funding_against | 197 | 36789.42 | 0.00033754 | 0.00162988 | -0.00162988 | 0.01579465 | 0.01416477 | 0.5330 | weak_or_insufficient |
| large_5d_inflow | 226 | 36532.69 | 0.00028221 | 0.00138741 | -0.00138741 | 0.01522096 | 0.01383354 | 0.5531 | weak_or_insufficient |
| btc_etf_inflow_context_label__funding_aligned | 29 | 13658.66 | 0.00000822 | 0.00021767 | 0.00030510 | 0.01218456 | 0.01248966 | 0.6897 | funding_regime_watch |
| large_5d_inflow__funding_aligned | 29 | 34788.76 | -0.00009362 | -0.00025967 | 0.00025967 | 0.01132379 | 0.01158346 | 0.6897 | funding_regime_watch |
| btc_etf_outflow_context_label__funding_aligned | 65 | -9592.45 | 0.00015116 | 0.00066122 | 0.00066167 | 0.01028185 | 0.01094352 | 0.5538 | funding_regime_watch |
| btc_etf_accumulation_label__funding_against | 211 | 30281.44 | 0.00029095 | 0.00145377 | -0.00145377 | 0.01121381 | 0.00976004 | 0.5498 | weak_or_insufficient |
| btc_etf_accumulation_label | 241 | 29708.39 | 0.00024150 | 0.00123905 | -0.00123905 | 0.01089986 | 0.00966081 | 0.5602 | weak_or_insufficient |
| btc_etf_accumulation_label__funding_aligned | 30 | 25677.97 | -0.00010633 | -0.00027111 | 0.00027111 | 0.00869181 | 0.00896292 | 0.6333 | funding_regime_watch |
| mixed_5d_flow__funding_aligned | 125 | -4596.83 | 0.00012081 | 0.00061086 | 0.00064102 | 0.00672348 | 0.00736450 | 0.5520 | funding_regime_watch |
| btc_etf_outflow_context_label | 86 | -8585.24 | 0.00012644 | 0.00062051 | 0.00036360 | 0.00560175 | 0.00596535 | 0.4535 | funding_regime_watch |
| mixed_5d_flow | 275 | 833.37 | 0.00014288 | 0.00073593 | -0.00017232 | 0.00613554 | 0.00596322 | 0.5418 | weak_or_insufficient |
| mixed_5d_flow__funding_against | 150 | 5358.55 | 0.00016127 | 0.00084015 | -0.00085010 | 0.00564558 | 0.00479549 | 0.5333 | weak_or_insufficient |
| btc_etf_distribution_label__funding_against | 13 | -22641.46 | -0.00009117 | -0.00012087 | -0.00012087 | 0.00281212 | 0.00269126 | 0.4615 | weak_or_insufficient |
| btc_etf_outflow_context_label__funding_against | 21 | -5467.71 | 0.00004990 | 0.00049449 | -0.00055900 | -0.00888425 | -0.00944324 | 0.1429 | weak_or_insufficient |

## Interpretation

The main tradable question is whether ETF flow direction and perp funding carry point the same way. For example, large ETF outflow plus positive BTCUSDT funding means the short BTC view also receives funding.
