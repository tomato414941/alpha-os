# Current Ticker Attention Source Split

This separates ticker-specific attention from broad market sentiment and duplicated news/event clusters before paper labels. It is not a trade instruction.

| symbol | source | specificity | decision | priority | context | event cluster | next probe |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| ZEC | coingecko_trending | ticker_mapped | dedupe_news_before_attention_label | 158.7744 | price=attention_price_lag_candidate; side=long_attention_lag; score=39.09753953 | two_source_event_pressure; sources=2; top=attention_price, rss:decrypt | label ZEC ticker attention separately from RSS/exchange event pressure |
| HYPE | coingecko_trending | ticker_mapped | ticker_specific_attention_alpha_candidate | 156.0226 | price=attention_price_lag_candidate; side=long_attention_lag; score=38.53373784 \|\| perp=attention_funding_watch; score=17.55665493 | two_source_event_pressure; sources=2; top=attention_market, attention_price | paper-label HYPE ticker-specific attention against price, funding, and depth |
| PENGU | coingecko_trending | ticker_mapped | ticker_specific_attention_alpha_candidate | 144.4629 | price=attention_price_lag_candidate; side=long_attention_lag; score=33.85161685 | single_event_context; sources=1; top=attention_price | paper-label PENGU ticker-specific attention against price, funding, and depth |
| ALLO | coingecko_trending | ticker_mapped | ticker_specific_attention_alpha_candidate | 143.5000 | price=attention_chase_risk; side=wait_or_fade_watch; score=42.00000000 | single_event_context; sources=1; top=attention_price | paper-label ALLO ticker-specific attention against price, funding, and depth |
| BTC | coingecko_trending | ticker_mapped | dedupe_news_before_attention_label | 143.4079 | price=attention_price_lag_candidate; side=long_attention_lag; score=25.63153287 | multi_source_event_pressure; sources=4; top=attention_price, rss:coindesk, rss:cointelegraph, rss:decrypt | label BTC ticker attention separately from RSS/exchange event pressure |
| BEAT | coingecko_trending | ticker_mapped | ticker_specific_attention_alpha_candidate | 138.6692 | price=attention_chase_risk; side=wait_or_fade_watch; score=34.67698556 | single_event_context; sources=1; top=attention_price | paper-label BEAT ticker-specific attention against price, funding, and depth |
| SOL | coingecko_trending | ticker_mapped | dedupe_news_before_attention_label | 132.2497 | price=attention_price_lag_candidate; side=long_attention_lag; score=32.99891434 | two_source_event_pressure; sources=2; top=attention_price, exchange_catalyst | label SOL ticker attention separately from RSS/exchange event pressure |
| TAO | coingecko_trending | ticker_mapped | ticker_specific_attention_alpha_candidate | 121.1444 | price=attention_price_lag_candidate; side=long_attention_lag; score=24.57760011 | single_event_context; sources=1; top=attention_price | paper-label TAO ticker-specific attention against price, funding, and depth |
| VVV | coingecko_trending | ticker_mapped | source_quality_required | 66.0000 | no joined market context | no event cluster | collect market context for VVV before treating attention as alpha |
| LIT | coingecko_trending | ticker_mapped | source_quality_required | 63.0000 | no joined market context | no event cluster | collect market context for LIT before treating attention as alpha |
| ONDO | coingecko_trending | ticker_mapped | source_quality_required | 57.0000 | no joined market context | no event cluster | collect market context for ONDO before treating attention as alpha |
| SERV | coingecko_trending | ticker_mapped | source_quality_required | 54.0000 | no joined market context | no event cluster | collect market context for SERV before treating attention as alpha |
| PI | coingecko_trending | ticker_mapped | source_quality_required | 51.0000 | no joined market context | no event cluster | collect market context for PI before treating attention as alpha |
| SUI | coingecko_trending | ticker_mapped | source_quality_required | 48.0000 | no joined market context | no event cluster | collect market context for SUI before treating attention as alpha |
| WLD | coingecko_trending | ticker_mapped | source_quality_required | 45.0000 | no joined market context | no event cluster | collect market context for WLD before treating attention as alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 19.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 18.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 17.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 16.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 15.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |

## Interpretation

`ticker_specific_attention_alpha_candidate` means the attention observation is mapped to a single ticker and has market context. `dedupe_news_before_attention_label` means the ticker attention is mixed with RSS or exchange-event pressure and must be separated before labeling. `broad_market_sentiment_control` is a control input, not a ticker alpha candidate.
