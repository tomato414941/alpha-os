# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| lane-near-near-microstructure-flow-paper-probe | NEAR | paper_long | 100.00 | 158.6563 | 1.83208904 | 30118.62350000 | 0.00332020 | -0.12500000 | 9.83208904 | 148.69921766 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| lane-near-near-microstructure-flow-probe | NEAR | paper_long | 100 | 158.6563 | 1.83208904 | 30118.62350000 | 0.00332020 | -0.12500000 | 9.83208904 | 148.69921766 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| lane-sol-sol-volume-price-dislocation | SOL | paper_long | 250 | 81.7215 | 0.15100608 | 391483.57990000 | 0.00063860 | 0.21253600 | 8.15100608 | 73.78300730 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| lane-hype-hype-protocol-fee-growth | HYPE | paper_short | 100 | 20.6152 | 0.81386169 | 45066.62538000 | 0.00221894 | 0.12500000 | 8.81386169 | 11.92634811 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| lane-hype-hype-unlock-actionability | HYPE | paper_short | 100 | 20.6152 | 0.81386169 | 45066.62538000 | 0.00221894 | 0.12500000 | 8.81386169 | 11.92634811 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| lane-eth-eth-volume-price-dislocation | ETH | paper_long | 250 | 16.2045 | 0.59925094 | 12978613.68000000 | 0.00001926 | -0.05781300 | 8.59925094 | 7.54747333 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| lane-eth-eth-microstructure-flow-probe | ETH | paper_long | 100 | 16.2045 | 0.59925094 | 12978613.68000000 | 0.00000770 | -0.05781300 | 8.59925094 | 7.54747333 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
