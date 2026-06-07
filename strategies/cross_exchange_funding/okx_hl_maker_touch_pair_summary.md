# OKX-Hyperliquid Maker Touch Pair Summary

This pairs OKX and Hyperliquid maker-touch observations by asset and sample window. Both legs must touch in the same window for a clean maker-maker entry proxy.

| asset | obs | both touch rate | either touch rate | OKX only | HL only | no touch | mean OKX edge bps | mean HL edge bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | 5 | 0.20000000 | 1.00000000 | 0.60000000 | 0.20000000 | 0.00000000 | 0.12224977 | 0.65889862 |
| BTC | 5 | 0.00000000 | 0.80000000 | 0.60000000 | 0.20000000 | 0.20000000 | 0.00807768 | 0.1452995 |

## Interpretation

A low both-touch rate means a maker-maker entry is unlikely to complete quickly without waiting, repricing, or crossing one leg. This still does not prove real fills because queue priority is unknown.
