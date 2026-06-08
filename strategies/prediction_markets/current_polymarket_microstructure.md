# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | New York Yankees vs. Cleveland Guardians | 0.6700 | 0.6800 | 0.0100 | 0.6750 | 0.1800 | 700632.68 | 63579.09 | 13.2987 | high activity and material one-day price move |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.8800 | 0.8900 | 0.0100 | 0.8850 | 0.0900 | 5683732.84 | 500722.13 | 13.1201 | high activity and material one-day price move |
| information_flow_watch | Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | 0.1170 | 0.1180 | 0.0010 | 0.1175 | -0.0955 | 5430121.00 | 290356.14 | 13.0997 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.1200 | 0.1300 | 0.0100 | 0.1250 | -0.1400 | 1511520.40 | 65333.70 | 13.0062 | high activity and material one-day price move |
| information_flow_watch | HSBC Championships: Katie Boulter vs Leylah Fernandez | 0.3000 | 0.3100 | 0.0100 | 0.3050 | -0.1800 | 400933.15 | 59844.12 | 12.9833 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. Tampa Bay Rays | 0.3700 | 0.3800 | 0.0100 | 0.3750 | -0.1400 | 335885.53 | 48027.28 | 12.0386 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 15, 2026? | 0.0500 | 0.0600 | 0.0100 | 0.0550 | -0.0300 | 2396970.81 | 606516.39 | 11.6269 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by end of June? | 0.0900 | 0.1000 | 0.0100 | 0.0950 | -0.0400 | 614403.20 | 505678.50 | 11.0410 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.2400 | 0.2500 | 0.0100 | 0.2450 | -0.0700 | 467095.51 | 44332.81 | 10.9053 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. Tampa Bay Rays: O/U 7.5 | 0.4400 | 0.4500 | 0.0100 | 0.4450 | -0.0800 | 345457.55 | 21341.05 | 10.6776 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.1100 | 0.1200 | 0.0100 | 0.1150 | -0.0600 | 250516.99 | 75914.75 | 10.4693 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 9? | 0.0260 | 0.0360 | 0.0100 | 0.0310 | -0.0505 | 354824.23 | 73540.11 | 10.4613 | high activity and material one-day price move |
| information_flow_watch | Will the U.S. invade Iran before 2027? | 0.1800 | 0.1900 | 0.0100 | 0.1850 | 0.0200 | 376065.94 | 424437.09 | 10.2955 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.2000 | 0.2100 | 0.0100 | 0.2050 | -0.0300 | 304175.30 | 127539.80 | 10.1801 | high activity and material one-day price move |
| information_flow_watch | Seattle Mariners vs. Baltimore Orioles | 0.4900 | 0.5000 | 0.0100 | 0.4950 | -0.0350 | 373709.05 | 52046.11 | 10.0139 | high activity and material one-day price move |
| information_flow_watch | Bab el-Mandeb Strait effectively closed by June 30? | 0.0950 | 0.1110 | 0.0160 | 0.1030 | 0.0300 | 239244.72 | 79834.74 | 9.8065 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
