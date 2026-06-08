# Binance Intraday Live Execution Gate

This checks the current execution side of Binance-derived intraday paper labels. Binance live feature endpoints may be unavailable by region, so OKX public book and funding are used for ARB perp execution context.

| symbol | feature | action | size | source | condition | spread | depth5 | slippage | funding1h | paper net | low-fee net | taker net | gate | reason |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ARBUSDT | count_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2160 | 4835 | 0.6080 | 0.0955 | 5.8666 | 4.1381 | -3.8619 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2160 | 4835 | 0.6080 | 0.0955 | 5.8666 | 4.1381 | -3.8619 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2160 | 4835 | 1.3169 | 0.0955 | 5.8666 | 3.4292 | -4.5708 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 250 | binance_region_blocked | unknown | 1.2160 | 4825 | 0.6080 | 0.0955 | 4.4252 | 2.6967 | -5.3033 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 100 | binance_region_blocked | unknown | 1.2160 | 4825 | 0.6080 | 0.0955 | 4.4252 | 2.6967 | -5.3033 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |
| ARBUSDT | count_top_long_short_ratio | short_opposite | 1000 | binance_region_blocked | unknown | 1.2160 | 4825 | 1.3286 | 0.0955 | 4.4252 | 1.9761 | -6.0239 | feature_source_blocked | Binance live feature endpoint is unavailable; execution context only |

## Interpretation

`low_fee_paper_probe` means the visible book does not obviously kill the low-cost paper edge. It still does not prove live alpha because the live Binance feature condition may be blocked, and maker fill probability, queue position, and stop behavior are unmodeled.
