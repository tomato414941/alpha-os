# Binance Intraday Live Execution Gate

This checks the current execution side of Binance-derived intraday paper labels. Binance live feature endpoints may be unavailable by region, so OKX public book and funding are used for ARB perp execution context.

| symbol | feature | action | size | source | condition | spread | depth5 | slippage | funding1h | paper net | low-fee net | taker net | gate | reason |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ARBUSDT | count_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2181 | 6933 | 1.7022 | 0.1210 | 5.8666 | 3.0673 | -4.9327 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2181 | 6933 | 1.7772 | 0.1210 | 5.8666 | 2.9923 | -5.0077 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2181 | 6933 | 2.1011 | 0.1210 | 5.8666 | 2.6684 | -5.3316 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2181 | 6069 | 1.7022 | 0.1210 | 4.4252 | 1.6259 | -6.3741 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2181 | 6069 | 1.7772 | 0.1210 | 4.4252 | 1.5509 | -6.4491 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2181 | 6069 | 2.1011 | 0.1210 | 4.4252 | 1.2269 | -6.7731 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |

## Interpretation

`low_fee_paper_probe` means the visible book does not obviously kill the low-cost paper edge. It still does not prove live alpha because the live Binance feature condition may be blocked, and maker fill probability, queue position, and stop behavior are unmodeled.
