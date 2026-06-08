# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 45.0000 | 1 | 35.38 | 122.26 | 287.95 | 3.1747 | avoid chasing ALLO; label pullback or fade setups instead |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 40.8084 | 6 | 5.56 | -22.02 | -29.69 | 0.1051 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| AAVE | Aave | attention_price_lag_candidate | long_attention_lag | 39.9622 | 8 | 2.20 | -21.43 | -33.55 | 0.1953 | paper-label AAVE attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 38.5898 | 3 | 6.57 | -16.26 | 40.53 | 0.0552 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 37.5325 | 2 | 76.16 | 312.87 | 791.47 | 0.0707 | avoid chasing BEAT; label pullback or fade setups instead |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 34.6874 | 7 | 3.24 | -18.35 | -29.32 | 0.0818 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 34.5222 | 5 | 3.29 | -10.23 | -35.21 | 0.2802 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| VELVET | Velvet | attention_chase_risk | wait_or_fade_watch | 32.5094 | 9 | 59.35 | 262.11 | 279.78 | 0.1102 | avoid chasing VELVET; label pullback or fade setups instead |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 28.0233 | 4 | 1.45 | -13.13 | -21.73 | 0.0288 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
