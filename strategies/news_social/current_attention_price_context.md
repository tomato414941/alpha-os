# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 49.8138 | 1 | 7.05 | -22.00 | -30.01 | 0.1554 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 37.4314 | 7 | 12.05 | -8.43 | 33.53 | 0.2340 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 37.0000 | 9 | 77.48 | 194.45 | 401.21 | 2.0981 | avoid chasing ALLO; label pullback or fade setups instead |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 36.6531 | 3 | 68.52 | 268.36 | 723.33 | 0.0731 | avoid chasing BEAT; label pullback or fade setups instead |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 35.4802 | 2 | 1.37 | -10.11 | -36.22 | 0.2815 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| BONK | Bonk | attention_price_lag_candidate | long_attention_lag | 34.4571 | 5 | 1.49 | -18.44 | -39.06 | 0.0704 | paper-label BONK attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 32.1157 | 6 | 4.38 | -14.91 | 39.78 | 0.0565 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 30.6033 | 10 | 2.27 | -18.40 | -29.09 | 0.0786 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 27.8839 | 4 | 1.18 | -13.33 | -21.34 | 0.0276 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
