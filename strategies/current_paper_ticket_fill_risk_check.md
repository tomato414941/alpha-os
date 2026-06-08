# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| paper-16-chip-repeat-execution | CHIP | paper_long | 1000 | 194.4264 | 3.17813444 | 4276.56800000 | 0.23383236 | 1.25039555 | 11.17813444 | 184.49870311 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| paper-05-mega-microstructure-flow | MEGA | paper_long | 100.00 | 182.9095 | 14.15146049 | 890.04241000 | 0.11235420 | -0.12500000 | 22.15146049 | 160.63305811 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| paper-03-inj-volume-dislocation | INJ | paper_long | 250 | 161.6753 |  |  |  | -0.00000000 | 8.00000000 | 153.67534722 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-27-pepe-liquidation-intensity | PEPE | paper_long | 100.00 | 111.8730 | 3.56824264 | 178519.16500000 | 0.00056016 | -0.62273098 | 11.56824264 | 99.68199643 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 77.7340 | 0.15100608 | 391483.57990000 | 0.00063860 | 0.21253600 | 8.15100608 | 69.79553056 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 76.3359 | 1.32899196 | 101153.63830000 | 0.00098860 | 0.25854524 | 9.32899196 | 67.26543114 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-28-doge-liquidation-intensity | DOGE | paper_long | 100.00 | 73.0707 |  |  |  | -0.00000000 | 8.00000000 | 65.07073294 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 66.6069 | 1.32927461 | 80606.36892000 | 0.01240597 | -0.12500000 | 9.32927461 | 57.15265091 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 66.6069 | 1.32927461 | 80606.36892000 | 0.00124060 | -0.12500000 | 9.32927461 | 57.15265091 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 65.5518 | 1.32899196 | 101153.63830000 | 0.00988595 | 0.25854524 | 9.32899196 | 56.48139274 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-12-virtual-volume-dislocation | VIRTUAL | paper_long | 250 | 53.4492 |  |  |  | -0.00000000 | 8.00000000 | 45.44919740 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-07-fartcoin-volume-dislocation | FARTCOIN | paper_long | 250 | 34.7373 |  |  |  | -0.00000000 | 8.00000000 | 26.73729917 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-18-near-microstructure-flow | NEAR | paper_long | 100.00 | 32.1632 | 1.83208904 | 30118.62350000 | 0.00332020 | -0.12500000 | 9.83208904 | 22.20611625 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-15-link-volume-dislocation | LINK | paper_long | 250 | 15.0637 |  |  |  | -0.00000000 | 8.00000000 | 7.06367250 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-24-zec-dislocation-repeat | ZEC | paper_short |  | 10.7509 |  |  |  | 0.00000000 | 8.00000000 | 2.75092902 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-50-hype-token-unlock | HYPE | paper_short |  | 5.3649 | 0.81386169 | 45066.62538000 |  | 0.12500000 | 8.81386169 | -3.32396761 | cost_adjusted_edge_failed | paper mark win does not survive rough spread, taker-fee, and funding haircut |
