# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-lane-sol-sol-volume-price-dislocation | SOL | paper_long | 250 | 245.1427 | 0.14812730 | 342298.14351500 | 0.00073036 | -0.12500000 | 8.14812730 | 236.86955637 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-sol-solana-stablecoin-migration | SOL | paper_long | 100 | 188.2321 | 0.14812730 | 342298.14351500 | 0.00029214 | -0.12500000 | 8.14812730 | 179.95897027 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-sol-sol-attention-price-context | SOL | paper_long | 100 | 188.2321 | 0.14812730 | 342298.14351500 | 0.00029214 | -0.12500000 | 8.14812730 | 179.95897027 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-eth-eth-volume-price-dislocation | ETH | paper_long | 250 | 176.8797 | 0.58908427 | 11054154.13066500 | 0.00002262 | -0.12500000 | 8.58908427 | 168.16563752 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-eth-eth-microstructure-flow-probe | ETH | paper_long | 100 | 176.8797 | 0.58908427 | 11054154.13066500 | 0.00000905 | -0.12500000 | 8.58908427 | 168.16563752 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-near-near-microstructure-flow-paper-probe | NEAR | paper_long | 100.00 | 142.1537 | 2.74674968 | 16877.98504000 | 0.00592488 | -0.12500000 | 10.74674968 | 131.28192495 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-near-near-microstructure-flow-probe | NEAR | paper_long | 100 | 119.5884 | 2.74674968 | 16877.98504000 | 0.00592488 | -0.12500000 | 10.74674968 | 108.71664376 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
