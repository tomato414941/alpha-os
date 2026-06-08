# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 44.9843 | 4 | 10.59 | -16.76 | -23.49 | 0.1245 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| H | Humanity | attention_capitulation_reversal_watch | watch_reversal_or_no_trade | 44.0000 | 2 | -55.08 | -45.32 | 54.24 | 0.5762 | wait for H reversal trigger, then label attention capitulation returns |
| WLD | Worldcoin | attention_chase_risk | wait_or_fade_watch | 40.0000 | 6 | 19.72 | 38.07 | 102.64 | 0.3321 | avoid chasing WLD; label pullback or fade setups instead |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 39.6072 | 10 | 11.36 | -13.61 | 39.78 | 0.2161 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 37.5696 | 3 | 5.13 | -9.44 | -35.13 | 0.2800 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| DEUS | XMAQUINA | attention_breakout_continuation_watch | long_momentum_watch | 37.5493 | 1 | 24.95 | 0.07 | 0.00 | 0.1495 | paper-label DEUS attention breakout continuation and stop behavior |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 33.1487 | 9 | 11.16 | -12.79 | 47.76 | 0.0671 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| ETH | Ethereum | attention_price_lag_candidate | long_attention_lag | 33.0254 | 7 | 4.54 | -15.15 | -27.54 | 0.0867 | paper-label ETH attention-price lag over 1h, 4h, 12h, and 24h |
| PIPPIN | pippin | attention_chase_risk | wait_or_fade_watch | 29.3828 | 5 | 83.43 | 41.91 | 4.88 | 1.0870 | avoid chasing PIPPIN; label pullback or fade setups instead |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
