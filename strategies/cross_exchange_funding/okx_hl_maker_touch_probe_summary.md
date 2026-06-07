# OKX-Hyperliquid Maker Touch Probe

This is a public-book proxy for maker feasibility. It places a virtual quote at the current best bid for buy legs or best ask for sell legs, then checks whether the next sampled opposite quote would cross it. It does not prove queue position or real fills.

| asset | venue | side | obs | touch rate | mean maker edge bps | min edge bps | max edge bps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ZEC | OkxSwap | buy | 5 | 0.60000000 | 0.12363111 | 0.12358037 | 0.12370573 |
| ZEC | HlPerp | sell | 5 | 0.20000000 | 0.59323956 | 0.12353152 | 1.23597172 |
| JTO | OkxSwap | buy | 5 | 0.80000000 | 0.77887741 | 0.77827068 | 0.78009205 |
| JTO | HlPerp | sell | 5 | 0.20000000 | 2.33828461 | 1.5579236 | 3.42695142 |
| BTC | OkxSwap | buy | 5 | 0.60000000 | 0.00810606 | 0.00810313 | 0.00810847 |
| BTC | HlPerp | sell | 5 | 0.40000000 | 0.08103426 | 0.08100774 | 0.08105633 |
| BABY | OkxSwap | sell | 5 | 0.20000000 | 3.22934853 | 3.22893122 | 3.23101777 |
| BABY | HlPerp | buy | 5 | 0.20000000 | 2.84211875 | 2.26061683 | 3.22913976 |

## Interpretation

Low touch rates mean the candidate may require waiting, repricing, or crossing the spread. High touch rates still do not prove maker fills because queue priority and post-only behavior are unknown.
