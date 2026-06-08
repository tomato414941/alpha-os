# Current Event Crypto Hedge Reaction Labels

This joins event-crypto hedge candidates to paper-ticket mark outcomes. It labels the market reaction after the candidate is opened; it is not a live PnL report.

| candidate | asset | action | reaction | elapsed min | entry | current | dir bps | event gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_hedge_reaction_pending | 0.01 | 63490.000000000000 | 63490.000000000000 |  | 0.175000 | 0.170000 | wait for the 15m checkpoint, then refresh marks and funding |
| eth_1962237_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_hedge_reaction_pending | 0.01 | 1687.400000000000 | 1687.400000000000 |  | 0.175000 | 0.170000 | wait for the 15m checkpoint, then refresh marks and funding |
| sol_1962237_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_hedge_reaction_pending | 0.01 | 67.469000000000 | 67.469000000000 |  | 0.175000 | 0.170000 | wait for the 15m checkpoint, then refresh marks and funding |
| btc_2296152_event_crypto_hedge | BTC | paper_short_risk_escalation | event_crypto_hedge_reaction_pending | 0.01 | 63490.000000000000 | 63490.000000000000 |  | 0.095000 | 0.090000 | wait for the 15m checkpoint, then refresh marks and funding |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 206.47 | 64034.000000000000 | 63490.000000000000 | -84.95486773 | 0.175000 | 0.170000 | record failure regime and check whether event odds were stale or non-causal |

## Summary

- event_crypto_hedge_reaction_loss: 1
- event_crypto_hedge_reaction_pending: 4
- best reaction: btc_1962237_event_crypto_hedge -84.95486773bps
- worst reaction: btc_1962237_event_crypto_hedge -84.95486773bps
