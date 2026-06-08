# Current Event Crypto Hedge Event Alignment

This checks whether the prediction-market event price moved with the crypto hedge return. It also compares the hedge return to same-asset non-event paper tickets. It is a rejection/control artifact, not a trade instruction.

| candidate | asset | status | asset bps | basket bps | event bps | controls | control mean | gap | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eth_1971905_event_crypto_hedge | ETH | event_probability_flat_crypto_moved | 99.46671460 | 76.66745814 | 0.00000000 | 4 | 99.46671460 | 0.00000000 | do not promote this event hedge; require event-market probability movement or stronger timestamp evidence |
| sol_1971905_event_crypto_hedge | SOL | event_probability_flat_crypto_moved | 71.24743388 | 76.66745814 | 0.00000000 | 1 | 71.24743388 | 0.00000000 | do not promote this event hedge; require event-market probability movement or stronger timestamp evidence |
| btc_1971905_event_crypto_hedge | BTC | event_probability_flat_crypto_moved | 59.28822593 | 76.66745814 | 0.00000000 | 1 | -59.28822593 | 118.57645186 | do not promote this event hedge; require event-market probability movement or stronger timestamp evidence |

## Summary

- event_probability_flat_crypto_moved: 3
