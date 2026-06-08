# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-lane-near-near-microstructure-flow-paper-probe | NEAR | paper_long | 100.00 | 143.0828 | 1.83208904 | 30118.62350000 | 0.00332020 | -0.12500000 | 9.83208904 | 133.12569457 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-near-near-microstructure-flow-probe | NEAR | paper_long | 100 | 120.5154 | 1.83208904 | 30118.62350000 | 0.00332020 | -0.12500000 | 9.83208904 | 110.55834621 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-sol-sol-volume-price-dislocation | SOL | paper_long | 250 | 55.8591 | 0.15100608 | 391483.57990000 | 0.00063860 | 0.21253600 | 8.15100608 | 47.92066775 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
