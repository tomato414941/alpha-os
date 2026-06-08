# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-repeat-paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 117.4392 | 1.05053052 | 51946.55373600 | 0.01925056 | -0.12500000 | 9.05053052 | 108.26365750 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 117.4392 | 1.05053052 | 51946.55373600 | 0.00192506 | -0.12500000 | 9.05053052 | 108.26365750 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 117.4392 | 1.05053052 | 51946.55373600 | 0.01925056 | -0.12500000 | 9.05053052 | 108.26365750 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 117.4392 | 1.05053052 | 51946.55373600 | 0.00192506 | -0.12500000 | 9.05053052 | 108.26365750 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 76.2667 | 0.14990144 | 286246.75024000 | 0.00087337 | 0.11966700 | 8.14990144 | 68.23647250 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
