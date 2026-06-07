# BTC ETF Flow Funding Intraday Entry Stress

This retests the BTC ETF-flow/funding paper rule with Binance BTCUSDT 1h closes. Entry is shifted by fixed hour offsets from the label-start UTC day and held for 120 hours. Funding PnL remains a rough daily approximation.

| group | trades | skipped | total return | mean net | hit | max drawdown | fee bps/side | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| entry_offset_hours_0 | 21 | 22 | 0.49268156 | 0.02101301 | 0.5714 | -0.10586191 | 5.0000 | survives |
| entry_offset_hours_8 | 21 | 22 | 0.57697523 | 0.02405146 | 0.6667 | -0.11468656 | 5.0000 | survives |
| entry_offset_hours_16 | 21 | 22 | 0.75997374 | 0.02891010 | 0.6667 | -0.06774742 | 5.0000 | survives |
| entry_offset_hours_24 | 21 | 22 | 0.69765587 | 0.02686266 | 0.6190 | -0.06945357 | 5.0000 | survives |
| entry_offset_hours_32 | 21 | 22 | 0.59012350 | 0.02399496 | 0.6667 | -0.12769816 | 5.0000 | survives |
| entry_offset_hours_48 | 20 | 23 | 0.25142354 | 0.01306676 | 0.6500 | -0.15900411 | 5.0000 | survives |

## Caveat

This is still not a live execution model. It does not simulate order books, liquidation, stop logic, or mark/index basis. Its purpose is only to test whether the daily close result is hypersensitive to entry timing.
