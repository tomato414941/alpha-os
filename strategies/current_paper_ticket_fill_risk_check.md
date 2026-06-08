# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| paper-05-mega-microstructure-flow | MEGA | paper_long | 100.00 | 413.2701 | 9.14476948 | 576.19495050 | 0.17355237 | 0.36800000 | 17.14476948 | 396.49329582 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| paper-16-chip-repeat-execution | CHIP | paper_long | 1000 | 249.5139 | 3.16105579 | 3171.89100000 | 0.31526935 | 0.13106834 | 11.16105579 | 238.48394645 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| paper-24-zec-dislocation-repeat | ZEC | paper_short |  | 46.9769 |  |  |  | 0.00000000 | 8.00000000 | 38.97688550 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 44.1945 | 1.33288904 | 94763.31120000 | 0.00105526 | 0.24782784 | 9.33288904 | 35.10939440 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-27-pepe-liquidation-intensity | PEPE | paper_long | 100.00 | 43.3057 | 3.59389039 | 143512.75600000 | 0.00069680 | -0.46588639 | 11.59389039 | 31.24588904 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-28-doge-liquidation-intensity | DOGE | paper_long | 100.00 | 41.3519 |  |  |  | -0.00000000 | 8.00000000 | 33.35192604 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 39.1884 | 1.19852182 | 52790.40296250 | 0.01894284 | -0.12500000 | 9.19852182 | 29.86489018 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 39.1884 | 1.19852182 | 52790.40296250 | 0.00189428 | -0.12500000 | 9.19852182 | 29.86489018 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 33.4448 | 1.33288904 | 94763.31120000 | 0.01055261 | 0.24782784 | 9.33288904 | 24.35975485 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 21.7534 | 0.15171859 | 401209.23253500 | 0.00062312 | 0.15687700 | 8.15171859 | 13.75850888 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-03-inj-volume-dislocation | INJ | paper_long | 250 | 6.6913 |  |  |  | -0.00000000 | 8.00000000 | -1.30873843 | missing_execution_context | no current public execution context for the promoted paper ticket |
