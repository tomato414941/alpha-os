# Current Event Crypto Hedge Reaction Labels

This joins event-crypto hedge candidates to paper-ticket mark outcomes. It labels the market reaction after the candidate is opened; it is not a live PnL report.

| candidate | asset | action | reaction | elapsed min | entry | current | dir bps | event gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sol_1971905_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_hedge_reaction_win | 149.74 | 66.248000000000 | 67.495000000000 | 188.23209757 | 0.335000 | 0.330000 | repeat with funding, spread/depth, beta attribution, and event timestamp controls |
| eth_1971905_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_hedge_reaction_win | 149.74 | 1668.900000000000 | 1697.300000000000 | 170.17196956 | 0.335000 | 0.330000 | repeat with funding, spread/depth, beta attribution, and event timestamp controls |
| btc_1971905_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_hedge_reaction_win | 149.74 | 63419.000000000000 | 64034.000000000000 | 96.97409294 | 0.335000 | 0.330000 | repeat with funding, spread/depth, beta attribution, and event timestamp controls |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_hedge_reaction_pending | 0.01 | 64034.000000000000 | 64034.000000000000 |  | 0.085000 | 0.080000 | wait for the 15m checkpoint, then refresh marks and funding |

## Summary

- event_crypto_hedge_reaction_pending: 1
- event_crypto_hedge_reaction_win: 3
- best reaction: sol_1971905_event_crypto_hedge 188.23209757bps
- worst reaction: btc_1971905_event_crypto_hedge 96.97409294bps
