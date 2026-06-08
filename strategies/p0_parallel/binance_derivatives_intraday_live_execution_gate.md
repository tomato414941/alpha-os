# Binance Intraday Live Execution Gate

This checks the current execution side of Binance-derived intraday paper labels. Binance live feature endpoints may be unavailable by region, so OKX public book and funding are used for ARB perp execution context.

| symbol | feature | action | size | source | condition | spread | depth5 | slippage | funding1h | paper net | low-fee net | taker net | gate | reason |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ARBUSDT | count_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2112 | 4773 | 0.6056 | 0.1250 | 5.8666 | 4.1749 | -3.8251 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2112 | 4773 | 0.7504 | 0.1250 | 5.8666 | 4.0300 | -3.9700 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2112 | 4949 | 0.6056 | 0.1250 | 4.4252 | 2.7334 | -5.2666 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2112 | 4949 | 0.6056 | 0.1250 | 4.4252 | 2.7334 | -5.2666 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2112 | 4773 | 2.1857 | 0.1250 | 5.8666 | 2.5947 | -5.4053 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2112 | 4949 | 1.7581 | 0.1250 | 4.4252 | 1.5809 | -6.4191 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |

## Interpretation

`low_fee_paper_probe` means the visible book does not obviously kill the low-cost paper edge. It still does not prove live alpha because the live Binance feature condition may be blocked, and maker fill probability, queue position, and stop behavior are unmodeled.
