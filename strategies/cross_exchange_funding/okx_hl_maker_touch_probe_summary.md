# OKX-Hyperliquid Maker Touch Probe

This is a public-book proxy for maker feasibility. It places a virtual quote at the current best bid for buy legs or best ask for sell legs, then checks whether the next sampled opposite quote would cross it. It does not prove queue position or real fills.

| asset | venue | side | obs | touch rate | mean maker edge bps | min edge bps | max edge bps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ZEC | HlPerp | sell | 5 | 0.60000000 | 0.517803 | 0.12306329 | 1.11071345 |
| ZEC | OkxSwap | buy | 5 | 0.40000000 | 0.12333502 | 0.12312087 | 0.12369961 |
| BTC | HlPerp | sell | 5 | 0.80000000 | 0.08094089 | 0.08086101 | 0.08099199 |
| BTC | OkxSwap | buy | 5 | 0.20000000 | 0.00809524 | 0.00808774 | 0.00809957 |

## Interpretation

Low touch rates mean the candidate may require waiting, repricing, or crossing the spread. High touch rates still do not prove maker fills because queue priority and post-only behavior are unknown.
