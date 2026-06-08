# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-lane-sol-sol-volume-price-dislocation | SOL | paper_long | 250 | 127.5046 | 0.14990144 | 286246.75024000 | 0.00087337 | 0.11966700 | 8.14990144 | 119.47431929 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-eth-eth-volume-price-dislocation | ETH | paper_long | 250 | 106.1278 | 1.18666192 | 10555414.30014000 | 0.00002368 | -0.08905600 | 9.18666192 | 96.85211515 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-eth-eth-microstructure-flow-probe | ETH | paper_long | 100 | 106.1278 | 1.18666192 | 10555414.30014000 | 0.00000947 | -0.08905600 | 9.18666192 | 96.85211515 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-sol-solana-stablecoin-migration | SOL | paper_long | 100 | 71.2474 | 0.14990144 | 286246.75024000 | 0.00034935 | 0.11966700 | 8.14990144 | 63.21719944 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-sol-sol-attention-price-context | SOL | paper_long | 100 | 71.2474 | 0.14990144 | 286246.75024000 | 0.00034935 | 0.11966700 | 8.14990144 | 63.21719944 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
