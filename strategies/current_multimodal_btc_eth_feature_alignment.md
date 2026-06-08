# Current Multimodal BTC/ETH Feature Alignment

This aligns NLP/news, ticker attention, stablecoin/on-chain proxy, wallet flow, funding market, and crypto-equity factor features for BTC and ETH. It is a feature alignment table, not a model or trade instruction.

| symbol | status | features | alignment | nlp | attention | stablecoin | wallet | funding | equity | boundary | next probe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ETH | multimodal_timestamp_boundary_required | 5 | 336.8354 | 66.7513 | 33.0641 | 45.9009 | 80.0000 | 0.0000 | 76.1192 | attention_news_dedupe,exchange_wallet_map_missing,equity_market_hours_gap | build ETH timestamp-aligned feature row before any multimodal label |
| BTC | multimodal_feature_label_priority | 5 | 238.3302 | 97.5790 | 8.3000 | 0.0000 | 55.7599 | 2.6493 | 14.0419 | aligned_timestamp_required | label BTC multimodal row with feature ablation and beta-adjusted target |

## Interpretation

A high alignment score means many feature families are present for the asset. It does not mean the features are causal or tradable. The next step is a leakage-safe timestamp table with feature ablation and beta-adjusted labels.
