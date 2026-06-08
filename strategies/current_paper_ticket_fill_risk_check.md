# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| paper-03-inj-volume-dislocation | INJ | paper_long | 250 | 191.5148 |  |  |  | -0.00000000 | 8.00000000 | 183.51475694 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-16-chip-repeat-execution | CHIP | paper_long | 1000 | 191.1860 | 3.18116749 | 4105.85500000 | 0.24355463 | 0.94476378 | 11.18116749 | 180.94959759 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| paper-05-mega-microstructure-flow | MEGA | paper_long | 100.00 | 185.3429 | 4.97755124 | 1663.36810900 | 0.06011898 | 0.00251100 | 12.97755124 | 172.36786441 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-27-pepe-liquidation-intensity | PEPE | paper_long | 100.00 | 111.8730 | 3.57079093 | 121857.52500000 | 0.00082063 | -0.57847132 | 11.57079093 | 99.72370780 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 83.0320 | 1.32828585 | 114099.57750000 | 0.00087643 | 0.25709960 | 9.32828585 | 73.96082125 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 80.1679 | 0.30189591 | 719563.91416000 | 0.00034743 | 0.13175600 | 8.30189591 | 71.99780204 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-28-doge-liquidation-intensity | DOGE | paper_long | 100.00 | 78.5922 |  |  |  | -0.00000000 | 8.00000000 | 70.59215488 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 72.8931 | 1.19522706 | 64449.36914650 | 0.01551606 | -0.12500000 | 9.19522706 | 63.57289424 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 72.8931 | 1.19522706 | 64449.36914650 | 0.00155161 | -0.12500000 | 9.19522706 | 63.57289424 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 72.2408 | 1.32828585 | 114099.57750000 | 0.00876427 | 0.25709960 | 9.32828585 | 63.16961643 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-24-zec-dislocation-repeat | ZEC | paper_short |  | 48.1455 |  |  |  | 0.00000000 | 8.00000000 | 40.14546474 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-12-virtual-volume-dislocation | VIRTUAL | paper_long | 250 | 45.8629 |  |  |  | -0.00000000 | 8.00000000 | 37.86285970 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-07-fartcoin-volume-dislocation | FARTCOIN | paper_long | 250 | 25.1845 |  |  |  | -0.00000000 | 8.00000000 | 17.18454190 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-20-mon-microstructure-flow | MON | paper_long | 100.00 | 17.8768 | 5.49123690 | 2463.11718900 | 0.04059896 | 0.00340800 | 13.49123690 | 4.38895878 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-50-hype-token-unlock | HYPE | paper_short |  | 13.9812 | 0.16273261 | 102246.87244500 |  | 0.12500000 | 8.16273261 | 5.94350652 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-15-link-volume-dislocation | LINK | paper_long | 250 | 5.0634 |  |  |  | -0.00000000 | 8.00000000 | -2.93658067 | missing_execution_context | no current public execution context for the promoted paper ticket |
