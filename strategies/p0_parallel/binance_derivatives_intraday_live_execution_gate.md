# Binance Intraday Live Execution Gate

This checks the current execution side of Binance-derived intraday paper labels. Binance live feature endpoints may be unavailable by region, so OKX public book and funding are used for ARB perp execution context.

| symbol | feature | action | size | source | condition | spread | depth5 | slippage | funding1h | paper net | low-fee net | taker net | gate | reason |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ARBUSDT | count_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2239 | 1940 | 2.8128 | 0.1181 | 5.8666 | 1.9480 | -6.0520 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2239 | 1940 | 3.0681 | 0.1181 | 5.8666 | 1.6927 | -6.3073 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2239 | 1940 | 3.9798 | 0.1181 | 5.8666 | 0.7810 | -7.2190 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2239 | 2200 | 2.8128 | 0.1181 | 4.4252 | 0.5065 | -7.4935 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2239 | 2200 | 3.0681 | 0.1181 | 4.4252 | 0.2513 | -7.7487 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2239 | 2200 | 3.9798 | 0.1181 | 4.4252 | -0.6604 | -8.6604 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |

## Interpretation

`low_fee_paper_probe` means the visible book does not obviously kill the low-cost paper edge. It still does not prove live alpha because the live Binance feature condition may be blocked, and maker fill probability, queue position, and stop behavior are unmodeled.
