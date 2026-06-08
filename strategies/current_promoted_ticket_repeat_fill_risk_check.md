# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 53.8752 | 0.15100608 | 391483.57990000 | 0.00063860 | 0.21253600 | 8.15100608 | 45.93675187 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 35.3333 | 1.32927461 | 80606.36892000 | 0.01240597 | -0.12500000 | 9.32927461 | 25.87905872 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 35.3333 | 1.32927461 | 80606.36892000 | 0.00124060 | -0.12500000 | 9.32927461 | 25.87905872 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 27.3115 | 1.32927461 | 80606.36892000 | 0.01240597 | -0.12500000 | 9.32927461 | 17.85720954 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 27.3115 | 1.32927461 | 80606.36892000 | 0.00124060 | -0.12500000 | 9.32927461 | 17.85720954 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
