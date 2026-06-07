# Event Flow Results

Generated on 2026-06-07 UTC.

## Sample

- Data source: Binance USD-M futures daily `aggTrades`
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Window: 2024-01-01 through 2024-01-03
- Bar size: 5 minutes

## First Diagnostic

| bucket | count | mean next 5m return | hit rate |
| --- | ---: | ---: | ---: |
| bottom_20 | 518 | 0.0000677538 | 0.530888 |
| middle_60 | 1553 | -0.0000287007 | 0.495815 |
| top_20 | 518 | -0.0000414307 | 0.500000 |

In this tiny sample, the highest positive taker-flow imbalance bucket does not
show a positive next-bar edge. That is not enough to close the lane. It only
means naive 5-minute imbalance needs a broader window, richer labels, and
execution-aware tests before it can be treated as a strategy candidate.

