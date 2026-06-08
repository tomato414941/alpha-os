# Binance Intraday Live Execution Gate

This checks the current execution side of Binance-derived intraday paper labels. Binance live feature endpoints may be unavailable by region, so OKX public book and funding are used for ARB perp execution context.

| symbol | feature | action | size | source | condition | spread | depth5 | slippage | funding1h | paper net | low-fee net | taker net | gate | reason |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ARBUSDT | count_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2191 | 3047 | 1.0148 | 0.0938 | 5.8666 | 3.7265 | -4.2735 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2191 | 3047 | 1.5031 | 0.0938 | 5.8666 | 3.2381 | -4.7619 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2193 | 4651 | 0.8890 | 0.0938 | 4.4252 | 2.4107 | -5.5893 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2191 | 3047 | 2.3885 | 0.0938 | 5.8666 | 2.3528 | -5.6472 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2193 | 4651 | 1.4530 | 0.0938 | 4.4252 | 1.8467 | -6.1533 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2193 | 4651 | 2.2970 | 0.0938 | 4.4252 | 1.0027 | -6.9973 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |

## Interpretation

`low_fee_paper_probe` means the visible book does not obviously kill the low-cost paper edge. It still does not prove live alpha because the live Binance feature condition may be blocked, and maker fill probability, queue position, and stop behavior are unmodeled.
