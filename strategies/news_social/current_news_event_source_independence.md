# Current News Event Source Independence

This checks whether multi-source news labels are actually independent stories. Multiple outlets repeating the same story is treated as weaker evidence than unrelated sources confirming the same direction. It is a control gate, not a trade instruction.

| symbol | kind | side | status | score | sources | labels | stories | dominant | mean 1h | mean 4h | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| ZEC | institutional_flow | paper_long_news_event | single_source_supported_story | 22.6523 | 1 | 1 | 1 | zec_security_vulnerability (1.00) | 749.15979224 | 226.09227009 | seek another independent ZEC source before treating the story as alpha |
| ZEC | narrative_event | paper_long_news_event | single_source_supported_story | 22.4286 | 1 | 1 | 1 | zec_security_vulnerability (1.00) | 85.71428571 | 632.85714286 | seek another independent ZEC source before treating the story as alpha |
| BTC | institutional_flow | paper_long_news_event | pending_archive_before_independence | 21.4344 | 3 | 9 | 2 | strategy_saylor_btc_treasury (0.89) | 118.33107131 | 134.71209543 | wait for fresh BTC archive labels before judging source independence |
| ZEC | security_risk | paper_short_news_event | single_source_supported_story | 10.3788 | 1 | 1 | 1 | zec_security_vulnerability (1.00) | 159.46951318 | -1380.23613133 | seek another independent ZEC source before treating the story as alpha |
| BTC | regulatory_risk | paper_short_news_event | reject_no_supported_label | -5.0000 | 1 | 1 | 1 | satoshi_era_bitcoin_center (1.00) | 0.00000000 | 0.00000000 | keep BTC news as context until source independence and forward labels improve |
| ETH | institutional_flow | paper_long_news_event | reject_no_supported_label | -5.0000 | 1 | 1 | 1 | tom_lee_s_ethereum (1.00) | 0.00000000 | 0.00000000 | keep ETH news as context until source independence and forward labels improve |
| USDC | narrative_event | paper_long_news_event | reject_no_supported_label | -5.0000 | 1 | 1 | 1 | travala_lets_ai_agents (1.00) | 0.00000000 | 0.00000000 | keep USDC news as context until source independence and forward labels improve |
| BTC | narrative_event | paper_long_news_event | reject_no_supported_label | -5.0000 | 1 | 1 | 1 | bitcoin_dumped_all_its (1.00) | 0.00000000 | 0.00000000 | keep BTC news as context until source independence and forward labels improve |

## Summary

- pending_archive_before_independence: 1
- reject_no_supported_label: 4
- single_source_supported_story: 3
- best source-independent candidate: ZEC/institutional_flow/paper_long_news_event status=single_source_supported_story sources=1 stories=1 score=22.65230675
