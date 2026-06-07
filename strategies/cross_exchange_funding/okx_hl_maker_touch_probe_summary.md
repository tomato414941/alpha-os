# OKX-Hyperliquid Maker Touch Probe

This is a public-book proxy for maker feasibility. It places a virtual quote at the current best bid for buy legs or best ask for sell legs, then checks whether the next sampled opposite quote would cross it. It does not prove queue position or real fills.

| asset | venue | side | obs | touch rate | mean maker edge bps | min edge bps | max edge bps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ZEC | OkxSwap | buy | 5 | 0.80000000 | 0.12224977 | 0.12205988 | 0.12232865 |
| ZEC | HlPerp | sell | 5 | 0.40000000 | 0.65889862 | 0.12219113 | 2.56125672 |
| BTC | OkxSwap | buy | 5 | 0.60000000 | 0.00807768 | 0.00807012 | 0.00808302 |
| BTC | HlPerp | sell | 5 | 0.20000000 | 0.1452995 | 0.08073175 | 0.32266391 |

## Interpretation

Low touch rates mean the candidate may require waiting, repricing, or crossing the spread. High touch rates still do not prove maker fills because queue priority and post-only behavior are unknown.
