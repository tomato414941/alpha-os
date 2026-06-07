# BTC ETF Flow Funding Candidate Rule

This is a non-overlapping paper rule for the large rolling ETF outflow plus start-funding-aligned BTC short candidate. It uses only label-start funding for entry filtering, then adds observed 5-day funding as rough PnL.

| rule | trades | skipped | total return | mean net 5d | hit 5d | max drawdown | fee bps/side | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| large_5d_outflow_start_funding_aligned_5d_hold | 21 | 22 | 0.78885853 | 0.02961324 | 0.6190 | -0.07880819 | 5.0000 | paper_rule_candidate |

## Robustness

| group | trades | total return | mean net 5d | hit 5d | max drawdown | action |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fee_bps_per_side_1.0 | 21 | 0.81836231 | 0.03041324 | 0.6667 | -0.07727182 | survives |
| fee_bps_per_side_5.0 | 21 | 0.78885853 | 0.02961324 | 0.6190 | -0.07880819 | survives |
| fee_bps_per_side_10.0 | 21 | 0.75261956 | 0.02861324 | 0.6190 | -0.08072685 | survives |
| fee_bps_per_side_20.0 | 21 | 0.68222803 | 0.02661324 | 0.6190 | -0.08455817 | survives |
| fee_bps_per_side_50.0 | 21 | 0.48684751 | 0.02061324 | 0.5714 | -0.09600412 | survives |
| entry_year_2024 | 3 | 0.13644321 | 0.04722467 | 0.6667 | -0.04913098 | thin_positive |
| entry_year_2025 | 10 | 0.21872627 | 0.02151472 | 0.5000 | -0.06087667 | thin_positive |
| entry_year_2026 | 8 | 0.29158230 | 0.03313210 | 0.7500 | -0.00372759 | survives |

## Caveat

This is not live-ready. It still ignores intraday fill timing, mark/index basis, liquidation buffer, and account-specific fees. Its value is that it removes overlapping signal inflation and stops using future funding as an entry condition.
