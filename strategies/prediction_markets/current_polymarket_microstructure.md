# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | New York Yankees vs. Cleveland Guardians | 0.7800 | 0.7900 | 0.0100 | 0.7850 | 0.2900 | 746542.07 | 27090.19 | 15.3480 | high activity and material one-day price move |
| information_flow_watch | Seattle Mariners vs. Baltimore Orioles | 0.3400 | 0.3500 | 0.0100 | 0.3450 | -0.1850 | 381444.32 | 48690.25 | 13.0106 | high activity and material one-day price move |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.8800 | 0.8900 | 0.0100 | 0.8850 | 0.0800 | 5689506.72 | 394562.55 | 12.8691 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.1200 | 0.1300 | 0.0100 | 0.1250 | -0.1300 | 1501012.36 | 59652.95 | 12.7834 | high activity and material one-day price move |
| information_flow_watch | HSBC Championships: Katie Boulter vs Leylah Fernandez | 0.3000 | 0.3100 | 0.0100 | 0.3050 | -0.1800 | 401341.29 | 23303.82 | 12.7790 | high activity and material one-day price move |
| information_flow_watch | Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | 0.1210 | 0.1240 | 0.0030 | 0.1225 | -0.0765 | 5536203.79 | 262957.85 | 12.7064 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 15, 2026? | 0.0500 | 0.0600 | 0.0100 | 0.0550 | -0.0300 | 2409466.68 | 623630.85 | 11.6353 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. Tampa Bay Rays | 0.4100 | 0.4200 | 0.0100 | 0.4150 | -0.1000 | 343980.59 | 106371.27 | 11.4242 | high activity and material one-day price move |
| information_flow_watch | Indiana Fever vs. Washington Mystics | 0.7900 | 0.8100 | 0.0200 | 0.8000 | 0.1050 | 313145.57 | 40304.27 | 11.2528 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by end of June? | 0.0900 | 0.1000 | 0.0100 | 0.0950 | -0.0400 | 605061.50 | 516388.67 | 11.0388 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 9? | 0.0300 | 0.0420 | 0.0120 | 0.0360 | -0.0685 | 360047.46 | 69195.27 | 10.8131 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.2400 | 0.2500 | 0.0100 | 0.2450 | -0.0650 | 457704.61 | 47038.90 | 10.8093 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.1800 | 0.1900 | 0.0100 | 0.1850 | -0.0600 | 301255.32 | 138005.58 | 10.7930 | high activity and material one-day price move |
| information_flow_watch | Will Vitality win IEM Cologne Major 2026? | 0.4400 | 0.4500 | 0.0100 | 0.4450 | -0.0200 | 632613.50 | 517800.81 | 10.5222 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.1100 | 0.1200 | 0.0100 | 0.1150 | -0.0600 | 250516.74 | 76942.99 | 10.4722 | high activity and material one-day price move |
| information_flow_watch | New York Yankees vs. Cleveland Guardians: O/U 7.5 | 0.6000 | 0.6300 | 0.0300 | 0.6150 | 0.0750 | 301742.78 | 13111.49 | 10.3784 | high activity and material one-day price move |
| information_flow_watch | Philadelphia Phillies vs. Toronto Blue Jays | 0.5700 | 0.5800 | 0.0100 | 0.5750 | -0.0500 | 374337.29 | 58633.12 | 10.3409 | high activity and material one-day price move |
| information_flow_watch | Will the U.S. invade Iran before 2027? | 0.1800 | 0.1900 | 0.0100 | 0.1850 | 0.0200 | 376045.18 | 420241.27 | 10.2933 | high activity and material one-day price move |
| information_flow_watch | Bab el-Mandeb Strait effectively closed by June 30? | 0.0960 | 0.1020 | 0.0060 | 0.0990 | 0.0280 | 238631.03 | 76417.91 | 9.7660 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
