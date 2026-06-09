# Current Event Crypto Hedge Reaction Labels

This joins event-crypto hedge candidates to paper-ticket mark outcomes. It labels the market reaction after the candidate is opened; it is not a live PnL report.

| candidate | asset | action | reaction | elapsed min | entry | current | dir bps | event gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| btc_2296152_event_crypto_hedge | BTC | paper_short_risk_escalation | event_crypto_hedge_reaction_win | 296.35 | 63490.000000000000 | 62473.000000000000 | 160.18270594 | 0.125000 | 0.120000 | repeat with funding, spread/depth, beta attribution, and event timestamp controls |
| eth_1971905_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 502.82 | 1697.300000000000 | 1663.500000000000 | -199.13981029 | 0.185000 | 0.180000 | record failure regime and check whether event odds were stale or non-causal |
| btc_1971905_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 502.82 | 64034.000000000000 | 62473.000000000000 | -243.77674360 | 0.185000 | 0.180000 | record failure regime and check whether event odds were stale or non-causal |
| sol_1971905_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 502.82 | 67.495000000000 | 65.463000000000 | -301.05933773 | 0.185000 | 0.180000 | record failure regime and check whether event odds were stale or non-causal |

## Summary

- event_crypto_hedge_reaction_loss: 3
- event_crypto_hedge_reaction_win: 1
- best reaction: btc_2296152_event_crypto_hedge 160.18270594bps
- worst reaction: sol_1971905_event_crypto_hedge -301.05933773bps
