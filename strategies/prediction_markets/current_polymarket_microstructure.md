# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | Seattle Mariners vs. Baltimore Orioles | 0.8900 | 0.9000 | 0.0100 | 0.8950 | 0.3600 | 415681.55 | 35069.49 | 16.4860 | high activity and material one-day price move |
| information_flow_watch | Philadelphia Phillies vs. Toronto Blue Jays | 0.9500 | 0.9600 | 0.0100 | 0.9550 | 0.3300 | 439628.17 | 67528.00 | 16.0588 | high activity and material one-day price move |
| information_flow_watch | New York Yankees vs. Cleveland Guardians: O/U 7.5 | 0.8700 | 0.8900 | 0.0200 | 0.8800 | 0.3400 | 302392.16 | 1999.37 | 15.2812 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. Tampa Bay Rays: O/U 7.5 | 0.2800 | 0.2900 | 0.0100 | 0.2850 | -0.2350 | 349771.71 | 18438.49 | 13.7526 | high activity and material one-day price move |
| information_flow_watch | Indiana Fever vs. Washington Mystics | 0.8600 | 0.8800 | 0.0200 | 0.8700 | 0.1950 | 317740.37 | 42871.59 | 13.0743 | high activity and material one-day price move |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.8700 | 0.8800 | 0.0100 | 0.8750 | 0.0800 | 5733650.74 | 330658.55 | 12.8347 | high activity and material one-day price move |
| information_flow_watch | HSBC Championships: Katie Boulter vs Leylah Fernandez | 0.3000 | 0.3100 | 0.0100 | 0.3050 | -0.1700 | 401460.06 | 50957.68 | 12.7491 | high activity and material one-day price move |
| information_flow_watch | Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | 0.1310 | 0.1350 | 0.0040 | 0.1330 | -0.0695 | 5632658.57 | 248441.62 | 12.5620 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 15, 2026? | 0.0500 | 0.0600 | 0.0100 | 0.0550 | -0.0300 | 2407738.63 | 618501.46 | 11.6331 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.1200 | 0.1300 | 0.0100 | 0.1250 | -0.0600 | 1480090.59 | 64168.10 | 11.3934 | high activity and material one-day price move |
| information_flow_watch | New York Yankees vs. Cleveland Guardians | 0.4200 | 0.4300 | 0.0100 | 0.4250 | -0.0700 | 768576.54 | 89260.39 | 11.2227 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by end of June? | 0.0900 | 0.1000 | 0.0100 | 0.0950 | -0.0300 | 597683.71 | 499289.84 | 10.8266 | high activity and material one-day price move |
| information_flow_watch | Will Vitality win IEM Cologne Major 2026? | 0.4300 | 0.4400 | 0.0100 | 0.4350 | -0.0300 | 634616.09 | 465488.52 | 10.7008 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.1800 | 0.1900 | 0.0100 | 0.1850 | -0.0500 | 305532.35 | 144480.25 | 10.6094 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.2400 | 0.2600 | 0.0200 | 0.2500 | -0.0500 | 456726.38 | 51230.64 | 10.5172 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.1100 | 0.1200 | 0.0100 | 0.1150 | -0.0600 | 246897.91 | 78314.74 | 10.4698 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. Tampa Bay Rays | 0.4600 | 0.4800 | 0.0200 | 0.4700 | -0.0450 | 370526.04 | 34838.76 | 10.1122 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 9? | 0.0220 | 0.0340 | 0.0120 | 0.0280 | -0.0305 | 352096.21 | 74999.01 | 10.0616 | high activity and material one-day price move |
| information_flow_watch | Bab el-Mandeb Strait effectively closed by June 30? | 0.0980 | 0.1120 | 0.0140 | 0.1050 | 0.0270 | 239752.75 | 75507.21 | 9.7374 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
