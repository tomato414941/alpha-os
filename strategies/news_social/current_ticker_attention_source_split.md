# Current Ticker Attention Source Split

This separates ticker-specific attention from broad market sentiment and duplicated news/event clusters before paper labels. It is not a trade instruction.

| symbol | source | specificity | decision | priority | context | event cluster | next probe |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| DEUS | coingecko_trending | ticker_mapped | ticker_specific_attention_alpha_candidate | 153.7717 | price=attention_breakout_continuation_watch; side=long_momentum_watch; score=47.08665576 | single_event_context; sources=1; top=attention_price | paper-label DEUS ticker-specific attention against price, funding, and depth |
| H | coingecko_trending | ticker_mapped | ticker_specific_attention_alpha_candidate | 150.0000 | price=attention_capitulation_reversal_watch; side=watch_reversal_or_no_trade; score=44.00000000 | single_event_context; sources=1; top=attention_price | paper-label H ticker-specific attention against price, funding, and depth |
| HYPE | coingecko_trending | ticker_mapped | dedupe_news_before_attention_label | 144.1837 | price=attention_price_lag_candidate; side=long_attention_lag; score=32.73478883 | two_source_event_pressure; sources=2; top=attention_price, rss:coindesk | label HYPE ticker attention separately from RSS/exchange event pressure |
| ZEC | coingecko_trending | ticker_mapped | dedupe_news_before_attention_label | 142.8734 | price=attention_price_lag_candidate; side=long_attention_lag; score=35.49373611 | two_source_event_pressure; sources=2; top=attention_price, rss:decrypt | label ZEC ticker attention separately from RSS/exchange event pressure |
| SOL | coingecko_trending | ticker_mapped | dedupe_news_before_attention_label | 138.6505 | price=attention_capitulation_reversal_watch; side=watch_reversal_or_no_trade; score=34.60188806 | two_source_event_pressure; sources=2; top=attention_price, exchange_catalyst | label SOL ticker attention separately from RSS/exchange event pressure |
| WLD | coingecko_trending | ticker_mapped | dedupe_news_before_attention_label | 136.0567 | price=attention_chase_risk; side=wait_or_fade_watch; score=36.22682841 | two_source_event_pressure; sources=2; top=attention_price, rss:cointelegraph | label WLD ticker attention separately from RSS/exchange event pressure |
| BEAT | coingecko_trending | ticker_mapped | ticker_specific_attention_alpha_candidate | 125.4265 | price=attention_chase_risk; side=wait_or_fade_watch; score=29.70618105 | single_event_context; sources=1; top=attention_price | paper-label BEAT ticker-specific attention against price, funding, and depth |
| BTC | coingecko_trending | ticker_mapped | source_quality_required | 101.0000 | no joined market context | multi_source_event_pressure; sources=3; top=rss:coindesk, rss:cointelegraph, rss:decrypt | collect market context for BTC before treating attention as alpha |
| PENGU | coingecko_trending | ticker_mapped | source_quality_required | 78.0000 | no joined market context | no event cluster | collect market context for PENGU before treating attention as alpha |
| VVV | coingecko_trending | ticker_mapped | source_quality_required | 60.0000 | no joined market context | no event cluster | collect market context for VVV before treating attention as alpha |
| SUI | coingecko_trending | ticker_mapped | source_quality_required | 57.0000 | no joined market context | no event cluster | collect market context for SUI before treating attention as alpha |
| ONDO | coingecko_trending | ticker_mapped | source_quality_required | 54.0000 | no joined market context | no event cluster | collect market context for ONDO before treating attention as alpha |
| XRP | coingecko_trending | ticker_mapped | source_quality_required | 51.0000 | no joined market context | no event cluster | collect market context for XRP before treating attention as alpha |
| LAB | coingecko_trending | ticker_mapped | source_quality_required | 48.0000 | no joined market context | no event cluster | collect market context for LAB before treating attention as alpha |
| LIT | coingecko_trending | ticker_mapped | source_quality_required | 45.0000 | no joined market context | no event cluster | collect market context for LIT before treating attention as alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 19.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 18.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 17.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 16.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |
| MARKET | alternative_me_fear_greed | broad_market | broad_market_sentiment_control | 15.0000 | no joined market context | no event cluster | keep fear/greed as market regime control, not as ticker paper alpha |

## Interpretation

`ticker_specific_attention_alpha_candidate` means the attention observation is mapped to a single ticker and has market context. `dedupe_news_before_attention_label` means the ticker attention is mixed with RSS or exchange-event pressure and must be separated before labeling. `broad_market_sentiment_control` is a control input, not a ticker alpha candidate.
