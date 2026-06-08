# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 56.3034 | 0.30189591 | 719563.91416000 | 0.00034743 | 0.13175600 | 8.30189591 | 48.13326106 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 41.6000 | 1.19522706 | 64449.36914650 | 0.00155161 | -0.12500000 | 9.19522706 | 32.27977294 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 41.6000 | 1.19522706 | 64449.36914650 | 0.01551606 | -0.12500000 | 9.19522706 | 32.27977294 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 33.5731 | 1.19522706 | 64449.36914650 | 0.01551606 | -0.12500000 | 9.19522706 | 24.25291443 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 33.5731 | 1.19522706 | 64449.36914650 | 0.00155161 | -0.12500000 | 9.19522706 | 24.25291443 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
