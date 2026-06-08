# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 50.2042 | 1 | 7.51 | -21.77 | -29.80 | 0.1539 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 37.6592 | 2 | 67.37 | 267.42 | 721.24 | 0.0732 | avoid chasing BEAT; label pullback or fade setups instead |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 37.0000 | 9 | 60.70 | 166.88 | 354.28 | 2.0442 | avoid chasing ALLO; label pullback or fade setups instead |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 36.4628 | 7 | 13.25 | -7.46 | 34.94 | 0.2320 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 33.7861 | 4 | 4.30 | -14.67 | 40.17 | 0.0563 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| BONK | Bonk | attention_price_lag_candidate | long_attention_lag | 33.3176 | 6 | 1.62 | -18.19 | -38.87 | 0.0701 | paper-label BONK attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 32.6215 | 5 | 1.74 | -9.88 | -36.06 | 0.2810 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 30.6101 | 10 | 2.30 | -18.36 | -29.05 | 0.0791 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 28.9755 | 3 | 1.37 | -13.23 | -21.25 | 0.0276 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
