# Current Multimodal BTC/ETH Feature Alignment

This aligns NLP/news, ticker attention, stablecoin/on-chain proxy, wallet flow, funding market, and crypto-equity factor features for BTC and ETH. It is a feature alignment table, not a model or trade instruction.

| symbol | status | features | alignment | nlp | attention | stablecoin | wallet | funding | equity | boundary | next probe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ETH | multimodal_timestamp_boundary_required | 4 | 285.5439 | 61.6061 | 0.0000 | 45.9189 | 80.0000 | 0.0000 | 75.0189 | stale_event,exchange_wallet_map_missing,equity_market_hours_gap | build ETH timestamp-aligned feature row before any multimodal label |
| BTC | multimodal_timestamp_boundary_required | 5 | 245.0352 | 99.5871 | 35.8520 | 0.0000 | 55.7599 | 4.6554 | 14.1808 | attention_news_dedupe | build BTC timestamp-aligned feature row before any multimodal label |

## Interpretation

A high alignment score means many feature families are present for the asset. It does not mean the features are causal or tradable. The next step is a leakage-safe timestamp table with feature ablation and beta-adjusted labels.
