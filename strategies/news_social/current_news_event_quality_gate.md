# Current News Event Quality Gate

This groups timestamped news-event labels by symbol, event kind, and side. It checks repeat support, source diversity, stale or pending labels, and rejected labels. It is a gate, not a trade instruction.

| symbol | kind | side | decision | score | sources | labels | support/reject/pending | mean 1h | mean 4h | best 4h | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| BTC | institutional_flow | paper_long_news_event | repeat_after_pending_archive | 39.9579 | 3 | 9 | 3/3/3 | 118.3311 | 134.7121 | 324.4693 | supported labels exist but fresh labels are still pending archive |
| ZEC | institutional_flow | paper_long_news_event | repeat_single_source_label | 34.7828 | 1 | 1 | 1/0/0 | 749.1598 | 226.0923 | 226.0923 | supported labels come from one source only |
| ZEC | narrative_event | paper_long_news_event | repeat_single_source_label | 34.2857 | 1 | 1 | 1/0/0 | 85.7143 | 632.8571 | 632.8571 | supported labels come from one source only |
| ZEC | security_risk | paper_short_news_event | repeat_single_source_label | 19.9735 | 1 | 1 | 1/0/0 | 159.4695 | -1380.2361 | -1380.2361 | supported labels come from one source only |
| BTC | regulatory_risk | paper_short_news_event | reject_no_supported_label | 4.0000 | 1 | 1 | 0/1/0 | 0.0000 | 0.0000 | 0.0000 | no supported timestamp label |
| ETH | institutional_flow | paper_long_news_event | reject_no_supported_label | 4.0000 | 1 | 1 | 0/1/0 | 0.0000 | 0.0000 | 0.0000 | no supported timestamp label |
| USDC | narrative_event | paper_long_news_event | reject_no_supported_label | 4.0000 | 1 | 1 | 0/1/0 | 0.0000 | 0.0000 | 0.0000 | no supported timestamp label |
| BTC | narrative_event | paper_long_news_event | reject_no_supported_label | 4.0000 | 1 | 1 | 0/1/0 | 0.0000 | 0.0000 | 0.0000 | no supported timestamp label |
