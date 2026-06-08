# Current News Event Screen

This classifies current crypto RSS headlines into event candidates and joins them to current perp state. It stores headline metadata only and is not a trade instruction.

| source | published | symbol | kind | status | side | score | funding | perp action | title |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| coindesk | 2026-06-08T18:41:04+00:00 | HYPE | macro_crypto | paper_news_macro_crypto_watch | risk_context | 55.1137 | 0.280210 | short_carry_reversion_watch | Influential research firm that caused AI stock meltdown lays out Hyperliquid as 'compelling' idea |
| decrypt | 2026-06-08T17:27:15+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 49.6146 | 0.000000 | - | For Bitcoin Giant Strategy, Cash Is Key to Calming Investors: JPMorgan |
| coindesk | 2026-06-08T17:04:38+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 49.2377 | 0.000000 | - | Live updates: Bitcoin tops $63,000 as Strategy adds $100 million BTC in latest purchase |
| cointelegraph | 2026-06-08T14:46:26+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 46.9343 | 0.000000 | - | Spot Bitcoin ETFs bleed $1.7B as outflow streak hits four weeks |
| coindesk | 2026-06-08T14:37:27+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 46.7846 | 0.000000 | - | Blame bitcoin's tumble on rising inflation, not Strategy, 10xResearch argues |
| cointelegraph | 2026-06-08T13:50:00+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 45.9938 | 0.000000 | - | Strategy buys 1,550 Bitcoin after controversial 32 BTC sale |
| decrypt | 2026-06-08T13:46:18+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 45.9321 | 0.000000 | - | Strategy Buys Bitcoin, Pads Cash Reserves Following Biggest Weekly Stock Drop Since 2022 |
| coindesk | 2026-06-08T12:12:45+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 44.3729 | 0.000000 | - | Strategy buys 1,550 bitcoin one week after selling $2.5 million of coins |
| decrypt | 2026-06-08T19:46:06+00:00 | BTC | narrative_event | paper_news_context_watch | collect_label | 38.9288 | 0.000000 | - | Bitcoin Is 'Boring' AI-Hungry Retail Investors, But Bernstein Still Sees $150K This Year |
| coindesk | 2026-06-08T05:08:06+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 37.2954 | 0.000000 | - | Bitcoin spikes, then dumps, from $63,700 as analysts assess Strategy's next BTC moves |
| decrypt | 2026-06-08T11:35:27+00:00 | ETH | stablecoin_event | paper_news_context_watch | collect_label | 35.7513 | 0.000000 | - | Reform UK's Farage 'Evading' Scrutiny Over Tether Billionaire's $6.7M Gift: Labour |
| decrypt | 2026-06-08T11:35:27+00:00 | USDT | stablecoin_event | paper_news_context_watch | collect_label | 35.7513 | 0.000000 | - | Reform UK's Farage 'Evading' Scrutiny Over Tether Billionaire's $6.7M Gift: Labour |
| cointelegraph | 2026-06-08T14:30:28+00:00 | BTC | narrative_event | paper_news_context_watch | collect_label | 33.6682 | 0.000000 | - | Bitcoin price eyes $90K as FTX-era BTC bullish divergence flashes again |
| coindesk | 2026-06-08T04:21:53+00:00 | BTC | macro_crypto | paper_news_macro_crypto_watch | risk_context | 32.5252 | 0.000000 | - | Bitcoin falls back below $63,000 as Iran-Israel trade strikes and Korean stocks crash |
| decrypt | 2026-06-08T12:41:22+00:00 | BTC | narrative_event | paper_news_context_watch | collect_label | 31.8499 | 0.000000 | - | Bitcoin's $63K Reclaim Liquidates $540M in Crypto Shorts, a 7-Week High |
| cointelegraph | 2026-06-07T19:45:17+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 28.0000 | 0.000000 | - | Strategy’s Saylor signals BTC buy as preferred dividend pay date vote looms |
| coindesk | 2026-06-07T17:41:57+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 28.0000 | 0.000000 | - | Michael Saylor revives bitcoin-buy speculation as scrutiny over Strategy grows |
| coindesk | 2026-06-07T16:14:08+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 28.0000 | 0.000000 | - | Bitcoin near $60,000 today vs February: Institutional sentiment has flipped |
| decrypt | 2026-06-05T20:02:53+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 28.0000 | 0.000000 | - | Strategy Shares Fall to 4-Month Low as STRC Dips and Bitcoin Sinks Under $60K |
| cointelegraph | 2026-06-07T20:52:36+00:00 | BTC | macro_crypto | paper_news_macro_crypto_watch | risk_context | 25.0371 | 0.000000 | - | What happens to Bitcoin if Nasdaq falls further? |
| decrypt | 2026-06-06T20:56:43+00:00 | ZEC | narrative_event | paper_news_event_reaction_watch | long_event_follow | 23.4879 | -0.249332 | long_carry_reversion_watch | AI Is Helping Discover Tech Vulnerabilities—And Zcash Is Just the Latest Example |
| cointelegraph | 2026-06-05T20:55:27+00:00 | USDC | narrative_event | paper_news_context_watch | collect_label | 15.0000 | 0.000000 | - | Travala lets AI agents book hotels with USDC on Base |
| decrypt | 2026-06-06T15:59:30+00:00 | BTC | narrative_event | paper_news_context_watch | collect_label | 15.0000 | 0.000000 | - | Bitcoin Has Dumped All of Its Gains Since Trump Was Reelected—And Then Some |

## Interpretation

News is a catalyst source, not an edge by itself. Rows need timestamp leakage checks, duplicate-source checks, venue depth, and forward-return labels before paper action.
