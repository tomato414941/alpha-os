# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-repeat-paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 229.4315 | 0.12987097 | 114777.07168950 | 0.00871254 | -0.12500000 | 8.12987097 | 221.17666491 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 229.4315 | 0.12987097 | 114777.07168950 | 0.00087125 | -0.12500000 | 8.12987097 | 221.17666491 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 229.4315 | 0.12987097 | 114777.07168950 | 0.00871254 | -0.12500000 | 8.12987097 | 221.17666491 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 229.4315 | 0.12987097 | 114777.07168950 | 0.00087125 | -0.12500000 | 8.12987097 | 221.17666491 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-22-bera-microstructure-flow | BERA | paper_long | 100.00 | 225.8758 | 6.38102209 | 643.52543250 | 0.15539401 | -0.12500000 | 14.38102209 | 211.36982302 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| repeat-repeat-paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 193.3097 | 0.14812730 | 342298.14351500 | 0.00073036 | -0.12500000 | 8.14812730 | 185.03654573 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-repeat-paper-21-chip-microstructure-flow | CHIP | paper_long | 100.00 | 15.6615 | 6.13214778 | 3766.24974000 | 0.02655161 | -0.12500000 | 14.13214778 | 1.40431887 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
