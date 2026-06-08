# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-paper-22-bera-microstructure-flow | BERA | paper_long | 100.00 | 437.8508 | 1.92060230 | 3981.40728900 | 0.02511675 | 0.02559100 | 9.92060230 | 427.95583072 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-21-chip-microstructure-flow | CHIP | paper_long | 100.00 | 345.0029 | 6.44893823 | 4900.41598800 | 0.02040643 | 0.00703900 | 14.44893823 | 330.56095991 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-20-mon-microstructure-flow | MON | paper_long | 100.00 | 242.5518 | 2.23358871 | 4984.66882700 | 0.02006151 | -0.12500000 | 10.23358871 | 232.19323958 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 154.2667 | 1.05053052 | 51946.55373600 | 0.01925056 | -0.12500000 | 9.05053052 | 145.09113615 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 154.2667 | 1.05053052 | 51946.55373600 | 0.00192506 | -0.12500000 | 9.05053052 | 145.09113615 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 146.1497 | 1.05053052 | 51946.55373600 | 0.01925056 | -0.12500000 | 9.05053052 | 136.97421635 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 146.1497 | 1.05053052 | 51946.55373600 | 0.00192506 | -0.12500000 | 9.05053052 | 136.97421635 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 125.5065 | 0.14990144 | 286246.75024000 | 0.00087337 | 0.11966700 | 8.14990144 | 117.47626853 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-paper-23-sei-microstructure-flow | SEI | paper_long | 100.00 | 4.4003 | 2.99889041 | 17207.91457350 | 0.00581128 | 0.17422500 | 10.99889041 | -6.42440139 | cost_adjusted_edge_failed | paper mark win does not survive rough spread, taker-fee, and funding haircut |
