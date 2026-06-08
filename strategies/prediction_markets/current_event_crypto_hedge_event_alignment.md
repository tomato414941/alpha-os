# Current Event Crypto Hedge Event Alignment

This checks whether the prediction-market event price moved with the crypto hedge return. It also compares the hedge return to same-asset non-event paper tickets. It is a rejection/control artifact, not a trade instruction.

| candidate | asset | status | asset bps | basket bps | event bps | controls | control mean | gap | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eth_1971905_event_crypto_hedge | ETH | event_alignment_inconclusive | -13.55093384 | -74.03812939 | -909.09090909 | 5 | -2.71018677 | -10.84074707 | collect a ready event-market ticket and same-asset controls before judging |
| sol_1971905_event_crypto_hedge | SOL | event_alignment_inconclusive | -85.19149567 | -74.03812939 | -909.09090909 | 4 | -78.50551046 | -6.68598521 | collect a ready event-market ticket and same-asset controls before judging |
| btc_1971905_event_crypto_hedge | BTC | event_alignment_inconclusive | -123.37195865 | -74.03812939 | -909.09090909 | 4 | -61.68597932 | -61.68597932 | collect a ready event-market ticket and same-asset controls before judging |
| eth_1962237_event_crypto_hedge | ETH | event_alignment_missing_event_ticket | 45.03970606 | -49.61230386 |  | 5 | -2.71018677 | 47.74989283 | collect a ready event-market ticket and same-asset controls before judging |
| btc_2296152_event_crypto_hedge | BTC | event_alignment_missing_event_ticket | 38.74625925 | 38.74625925 |  | 4 | -61.68597932 | 100.43223857 | collect a ready event-market ticket and same-asset controls before judging |
| btc_1962237_event_crypto_hedge | BTC | event_alignment_missing_event_ticket | -38.74625925 | -49.61230386 |  | 4 | -61.68597932 | 22.93972007 | collect a ready event-market ticket and same-asset controls before judging |
| sol_1962237_event_crypto_hedge | SOL | event_alignment_missing_event_ticket | -81.37070358 | -49.61230386 |  | 4 | -78.50551046 | -2.86519312 | collect a ready event-market ticket and same-asset controls before judging |
| btc_1962237_event_crypto_hedge | BTC | event_alignment_missing_event_ticket | -123.37195865 | -49.61230386 |  | 4 | -61.68597932 | -61.68597932 | collect a ready event-market ticket and same-asset controls before judging |

## Summary

- event_alignment_inconclusive: 3
- event_alignment_missing_event_ticket: 5
