# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H | Humanity | attention_capitulation_reversal_watch | watch_reversal_or_no_trade | 44.0000 | 2 | -68.76 | -62.64 | 11.73 | 0.9491 | wait for H reversal trigger, then label attention capitulation returns |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 43.6532 | 4 | 10.98 | -15.78 | -21.86 | 0.1175 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| DEUS | XMAQUINA | attention_breakout_continuation_watch | long_momentum_watch | 40.2991 | 1 | 24.94 | 1.82 | 0.00 | 0.1696 | paper-label DEUS attention breakout continuation and stop behavior |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 39.9081 | 10 | 8.47 | -15.44 | 39.60 | 0.2163 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| WLD | Worldcoin | attention_chase_risk | wait_or_fade_watch | 39.8803 | 6 | 16.16 | 34.61 | 99.40 | 0.3408 | avoid chasing WLD; label pullback or fade setups instead |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 35.1536 | 3 | 3.29 | -8.87 | -35.41 | 0.2780 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 32.7498 | 9 | 9.27 | -13.13 | 48.15 | 0.0671 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| ETH | Ethereum | attention_price_lag_candidate | long_attention_lag | 32.4215 | 7 | 3.73 | -15.32 | -27.61 | 0.0875 | paper-label ETH attention-price lag over 1h, 4h, 12h, and 24h |
| PIPPIN | pippin | attention_chase_risk | wait_or_fade_watch | 28.1748 | 5 | 65.79 | 35.87 | -0.76 | 1.1730 | avoid chasing PIPPIN; label pullback or fade setups instead |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
