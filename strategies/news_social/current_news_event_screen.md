# Current News Event Screen

This classifies current crypto RSS headlines into event candidates and joins them to current perp state. It stores headline metadata only and is not a trade instruction.

| source | published | symbol | kind | status | side | score | funding | perp action | title |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| cointelegraph | 2026-06-07T19:45:17+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 46.6370 | 0.000000 | - | Strategy’s Saylor signals BTC buy as preferred dividend pay date vote looms |
| coindesk | 2026-06-07T17:41:57+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 44.5815 | 0.000000 | - | Michael Saylor revives bitcoin-buy speculation as scrutiny over Strategy grows |
| cointelegraph | 2026-06-07T20:52:36+00:00 | BTC | macro_crypto | paper_news_macro_crypto_watch | risk_context | 43.7590 | 0.000000 | - | What happens to Bitcoin if the Nasdaq falls further? |
| coindesk | 2026-06-07T16:14:08+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 43.1178 | 0.000000 | - | Bitcoin near $60,000 today vs February: Institutional sentiment has flipped |
| coindesk | 2026-06-07T16:00:00+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 42.8823 | 0.000000 | - | Bitcoin's slide has no single cause. AI, tech IPOs, quantum, Strategy sale all play a role, NYDIG says |
| decrypt | 2026-06-05T12:49:04+00:00 | ZEC | security_risk | paper_news_security_risk_watch | short_or_avoid | 41.1586 | -0.606761 | long_carry_reversion_watch | Morning Minute: Massive ZCash Exploit Found by Claude, Extent Unknown |
| decrypt | 2026-06-05T15:49:08+00:00 | ZEC | institutional_flow | paper_news_event_reaction_watch | long_event_follow | 37.1586 | -0.606761 | long_carry_reversion_watch | Winklevoss-Backed Zcash Treasury Plunges Nearly 40% on ZEC Privacy Bug Concerns |
| coindesk | 2026-06-06T13:30:00+00:00 | BTC | regulatory_risk | paper_news_regulatory_risk_watch | short_or_avoid | 30.0000 | 0.000000 | - | Satoshi-era bitcoin at center of $285 billion lawsuit moves after 14 years |
| cointelegraph | 2026-06-05T07:00:55+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 28.0000 | 0.000000 | - | Strategy’s leveraged Bitcoin model has faced its first stress test: Grayscale |
| decrypt | 2026-06-05T20:02:53+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 28.0000 | 0.000000 | - | Strategy Shares Fall to 4-Month Low as STRC Dips and Bitcoin Sinks Under $60K |
| decrypt | 2026-06-05T17:57:29+00:00 | ETH | institutional_flow | paper_news_context_watch | collect_label | 28.0000 | 0.000000 | - | Tom Lee's Ethereum Treasury BitMine Prices Preferred Shares With 9.5% Dividend |
| decrypt | 2026-06-05T16:15:02+00:00 | BTC | institutional_flow | paper_news_context_watch | collect_label | 28.0000 | 0.000000 | - | What Is Strategy (MSTR)? The Bitcoin Treasury Company |
| coindesk | 2026-06-06T09:38:00+00:00 | ZEC | narrative_event | paper_news_event_reaction_watch | long_event_follow | 24.1586 | -0.606761 | long_carry_reversion_watch | Researcher who found Zcash's bug with AI adds Monero to his audit queue |
| decrypt | 2026-06-06T20:56:43+00:00 | ZEC | narrative_event | paper_news_event_reaction_watch | long_event_follow | 24.1586 | -0.606761 | long_carry_reversion_watch | AI Is Helping Discover Tech Vulnerabilities—And Zcash Is Just the Latest Example |
| coindesk | 2026-06-06T15:00:00+00:00 | BTC | macro_crypto | paper_news_macro_crypto_watch | risk_context | 24.0000 | 0.000000 | - | A crypto pioneer who turned a $20 million family stake into a billion-dollar fund doubles down on bitcoin |
| cointelegraph | 2026-06-05T10:59:50+00:00 | SOL | narrative_event | paper_news_event_reaction_watch | long_event_follow | 23.2519 | -0.126989 | long_carry_reversion_watch | Forward Industries moves $32M in SOL amid $1B paper loss |
| cointelegraph | 2026-06-05T20:55:27+00:00 | USDC | narrative_event | paper_news_context_watch | collect_label | 15.0000 | 0.000000 | - | Travala lets AI agents book hotels with USDC on Base |
| cointelegraph | 2026-06-05T12:24:23+00:00 | BTC | narrative_event | paper_news_context_watch | collect_label | 15.0000 | 0.000000 | - | How low can Bitcoin price go if $60K support fails? |
| coindesk | 2026-06-06T09:45:15+00:00 | BTC | narrative_event | paper_news_context_watch | collect_label | 15.0000 | 0.000000 | - | Are retail traders selling their bitcoin to buy the SpaceX IPO? |
| coindesk | 2026-06-05T16:01:00+00:00 | BTC | narrative_event | paper_news_context_watch | collect_label | 15.0000 | 0.000000 | - | Memecoins dogecoin, shiba inu dive 9% as bitcoin nears $60,000 |
| coindesk | 2026-06-05T16:01:00+00:00 | DOGE | narrative_event | paper_news_context_watch | collect_label | 15.0000 | 0.000000 | - | Memecoins dogecoin, shiba inu dive 9% as bitcoin nears $60,000 |
| decrypt | 2026-06-06T15:59:30+00:00 | BTC | narrative_event | paper_news_context_watch | collect_label | 15.0000 | 0.000000 | - | Bitcoin Has Dumped All of Its Gains Since Trump Was Reelected—And Then Some |

## Interpretation

News is a catalyst source, not an edge by itself. Rows need timestamp leakage checks, duplicate-source checks, venue depth, and forward-return labels before paper action.
