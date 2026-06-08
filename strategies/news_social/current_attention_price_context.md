# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 42.0000 | 4 | 45.20 | 157.57 | 323.34 | 3.6623 | avoid chasing ALLO; label pullback or fade setups instead |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 39.0975 | 2 | 1.55 | -17.49 | -25.35 | 0.1211 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 38.5337 | 1 | 10.36 | -10.45 | 48.27 | 0.0617 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 34.6770 | 5 | 59.37 | 300.22 | 753.14 | 0.0735 | avoid chasing BEAT; label pullback or fade setups instead |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 33.8516 | 3 | 3.41 | -7.44 | -34.18 | 0.2796 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 32.9989 | 7 | 4.01 | -15.62 | -27.87 | 0.0874 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 25.6315 | 6 | 3.63 | -10.53 | -20.33 | 0.0294 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |
| TAO | Bittensor | attention_price_lag_candidate | long_attention_lag | 24.5776 | 10 | 2.11 | -12.39 | -29.99 | 0.0817 | paper-label TAO attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
