# Current Event Crypto Hedge Reaction Labels

This joins event-crypto hedge candidates to paper-ticket mark outcomes. It labels the market reaction after the candidate is opened; it is not a live PnL report.

| candidate | asset | action | reaction | elapsed min | entry | current | dir bps | event gap | edge | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eth_1962237_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_hedge_reaction_win | 175.58 | 1687.400000000000 | 1695.000000000000 | 45.03970606 | 0.175000 | 0.170000 | repeat with funding, spread/depth, beta attribution, and event timestamp controls |
| btc_2296152_event_crypto_hedge | BTC | paper_short_risk_escalation | event_crypto_hedge_reaction_win | 175.58 | 63490.000000000000 | 63244.000000000000 | 38.74625925 | 0.085000 | 0.080000 | repeat with funding, spread/depth, beta attribution, and event timestamp controls |
| eth_1971905_event_crypto_hedge | ETH | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 382.04 | 1697.300000000000 | 1695.000000000000 | -13.55093384 | 0.185000 | 0.180000 | record failure regime and check whether event odds were stale or non-causal |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 175.58 | 63490.000000000000 | 63244.000000000000 | -38.74625925 | 0.175000 | 0.170000 | record failure regime and check whether event odds were stale or non-causal |
| sol_1962237_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 175.58 | 67.469000000000 | 66.920000000000 | -81.37070358 | 0.175000 | 0.170000 | record failure regime and check whether event odds were stale or non-causal |
| sol_1971905_event_crypto_hedge | SOL | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 382.04 | 67.495000000000 | 66.920000000000 | -85.19149567 | 0.185000 | 0.180000 | record failure regime and check whether event odds were stale or non-causal |
| btc_1971905_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 382.04 | 64034.000000000000 | 63244.000000000000 | -123.37195865 | 0.185000 | 0.180000 | record failure regime and check whether event odds were stale or non-causal |
| btc_1962237_event_crypto_hedge | BTC | paper_long_risk_relief | event_crypto_hedge_reaction_loss | 382.04 | 64034.000000000000 | 63244.000000000000 | -123.37195865 | 0.175000 | 0.170000 | record failure regime and check whether event odds were stale or non-causal |

## Summary

- event_crypto_hedge_reaction_loss: 6
- event_crypto_hedge_reaction_win: 2
- best reaction: eth_1962237_event_crypto_hedge 45.03970606bps
- worst reaction: btc_1971905_event_crypto_hedge -123.37195865bps
