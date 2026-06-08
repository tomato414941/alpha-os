# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 48.8671 | 0.15102888 | 691488.91950000 | 0.00036154 | 0.13408400 | 8.15102888 | 40.85015785 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 36.4000 | 1.46168719 | 52473.70350700 | 0.00190572 | -0.12500000 | 9.46168719 | 26.81331281 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 36.4000 | 1.46168719 | 52473.70350700 | 0.01905716 | -0.12500000 | 9.46168719 | 26.81331281 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 28.3773 | 1.46168719 | 52473.70350700 | 0.01905716 | -0.12500000 | 9.46168719 | 18.79061097 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 28.3773 | 1.46168719 | 52473.70350700 | 0.00190572 | -0.12500000 | 9.46168719 | 18.79061097 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
