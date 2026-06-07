# BTC ETF Flow Funding Candidate Rule

This is a non-overlapping paper rule for the large rolling ETF outflow plus start-funding-aligned BTC short candidate. It uses only label-start funding for entry filtering, then adds observed 5-day funding as rough PnL.

| rule | trades | skipped | total return | mean net 5d | hit 5d | max drawdown | fee bps/side | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| large_5d_outflow_start_funding_aligned_5d_hold | 21 | 22 | 0.78885853 | 0.02961324 | 0.6190 | -0.07880819 | 5.0000 | paper_rule_candidate |

## Caveat

This is not live-ready. It still ignores intraday fill timing, mark/index basis, liquidation buffer, and account-specific fees. Its value is that it removes overlapping signal inflation and stops using future funding as an entry condition.
