# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 44.0000 | 2 | 44.96 | 136.65 | 313.06 | 3.4558 | avoid chasing ALLO; label pullback or fade setups instead |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 40.5858 | 8 | 5.39 | -22.51 | -30.13 | 0.1440 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 38.5184 | 1 | 71.95 | 300.04 | 763.78 | 0.0704 | avoid chasing BEAT; label pullback or fade setups instead |
| AAVE | Aave | attention_price_lag_candidate | long_attention_lag | 38.1913 | 10 | 2.78 | -21.17 | -33.33 | 0.1882 | paper-label AAVE attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 37.4551 | 4 | 6.68 | -16.02 | 40.93 | 0.0551 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 37.1657 | 3 | 4.29 | -9.88 | -34.95 | 0.2786 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 35.7993 | 6 | 3.55 | -18.15 | -29.14 | 0.0818 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 34.9536 | 9 | 14.45 | -7.95 | 35.41 | 0.2521 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 27.3869 | 5 | 2.05 | -12.91 | -21.53 | 0.0287 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
