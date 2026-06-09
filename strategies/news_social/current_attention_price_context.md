# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DEUS | XMAQUINA | attention_breakout_continuation_watch | long_momentum_watch | 47.0867 | 1 | 28.64 | 8.22 | 0.00 | 0.1774 | paper-label DEUS attention breakout continuation and stop behavior |
| H | Humanity | attention_capitulation_reversal_watch | watch_reversal_or_no_trade | 44.0000 | 2 | -89.84 | -89.13 | -64.05 | 3.5189 | wait for H reversal trigger, then label attention capitulation returns |
| WLD | Worldcoin | attention_chase_risk | wait_or_fade_watch | 36.2268 | 6 | 2.29 | 14.14 | 81.13 | 0.4378 | avoid chasing WLD; label pullback or fade setups instead |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 35.4937 | 7 | 2.96 | -17.20 | -24.50 | 0.1267 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_capitulation_reversal_watch | watch_reversal_or_no_trade | 34.6019 | 5 | -2.07 | -19.34 | -29.89 | 0.0852 | wait for SOL reversal trigger, then label attention capitulation returns |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 32.7348 | 3 | 1.21 | -15.20 | 44.86 | 0.0665 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 29.7062 | 9 | 18.65 | 276.41 | 789.64 | 0.0541 | avoid chasing BEAT; label pullback or fade setups instead |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
