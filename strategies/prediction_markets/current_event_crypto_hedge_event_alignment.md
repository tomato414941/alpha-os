# Current Event Crypto Hedge Event Alignment

This checks whether the prediction-market event price moved with the crypto hedge return. It also compares the hedge return to same-asset non-event paper tickets. It is a rejection/control artifact, not a trade instruction.

| candidate | asset | status | asset bps | basket bps | event bps | controls | control mean | gap | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| btc_1962237_event_crypto_hedge | BTC | event_alignment_missing_event_ticket |  |  |  | 6 | -56.63657849 | 56.63657849 | collect a ready event-market ticket and same-asset controls before judging |
| eth_1962237_event_crypto_hedge | ETH | event_alignment_missing_event_ticket |  |  |  | 7 | -24.99768540 | 24.99768540 | collect a ready event-market ticket and same-asset controls before judging |
| sol_1962237_event_crypto_hedge | SOL | event_alignment_missing_event_ticket |  |  |  | 5 | -3.85213720 | 3.85213720 | collect a ready event-market ticket and same-asset controls before judging |
| btc_2296152_event_crypto_hedge | BTC | event_alignment_missing_event_ticket |  |  |  | 6 | -56.63657849 | 56.63657849 | collect a ready event-market ticket and same-asset controls before judging |
| btc_1962237_event_crypto_hedge | BTC | event_alignment_missing_event_ticket | -84.95486773 | -84.95486773 |  | 6 | -56.63657849 | -28.31828924 | collect a ready event-market ticket and same-asset controls before judging |

## Summary

- event_alignment_missing_event_ticket: 5
