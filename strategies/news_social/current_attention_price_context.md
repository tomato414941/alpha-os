# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 51.3043 | 3 | 26.46 | -19.24 | -25.37 | 0.1812 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 46.7453 | 1 | 10.81 | -11.75 | -33.73 | 0.2910 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 42.5441 | 4 | 7.68 | -18.84 | -27.97 | 0.0805 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| ETH | Ethereum | attention_price_lag_candidate | long_attention_lag | 39.0901 | 5 | 8.07 | -16.09 | -27.09 | 0.0785 | paper-label ETH attention-price lag over 1h, 4h, 12h, and 24h |
| TAO | Bittensor | attention_price_lag_candidate | long_attention_lag | 37.8358 | 9 | 10.38 | -16.04 | -32.06 | 0.0959 | paper-label TAO attention-price lag over 1h, 4h, 12h, and 24h |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 36.8765 | 8 | 10.22 | -8.88 | 29.97 | 0.2434 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 33.9743 | 2 | 4.23 | -14.31 | -21.35 | 0.0286 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 33.1157 | 6 | 5.61 | -14.93 | 37.68 | 0.0515 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| WLD | Worldcoin | attention_chase_risk | wait_or_fade_watch | 31.3632 | 10 | 16.50 | 43.31 | 76.82 | 0.3850 | avoid chasing WLD; label pullback or fade setups instead |
| XRP | XRP | attention_price_lag_candidate | long_attention_lag | 30.0615 | 7 | 6.62 | -12.98 | -18.12 | 0.0292 | paper-label XRP attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
