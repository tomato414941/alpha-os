# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | New York Yankees vs. Cleveland Guardians: O/U 7.5 | 0.9900 | 1.0000 | 0.0100 | 0.9950 | 0.4550 | 304017.38 | 25104.80 | 18.1435 | high activity and material one-day price move |
| information_flow_watch | Seattle Mariners vs. Baltimore Orioles | 0.9400 | 0.9500 | 0.0100 | 0.9450 | 0.4100 | 525257.85 | 40113.68 | 17.6422 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. Tampa Bay Rays: O/U 7.5 | 0.1200 | 0.1400 | 0.0200 | 0.1300 | -0.3850 | 352829.92 | 9302.08 | 16.5988 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. Tampa Bay Rays | 0.1900 | 0.2000 | 0.0100 | 0.1950 | -0.3200 | 440113.62 | 50484.60 | 15.7965 | high activity and material one-day price move |
| information_flow_watch | Philadelphia Phillies vs. Toronto Blue Jays | 0.9000 | 0.9100 | 0.0100 | 0.9050 | 0.2800 | 504667.00 | 44091.46 | 15.0411 | high activity and material one-day price move |
| information_flow_watch | New York Yankees vs. Cleveland Guardians | 0.2600 | 0.2700 | 0.0100 | 0.2650 | -0.2300 | 845509.49 | 38244.75 | 14.2905 | high activity and material one-day price move |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.8600 | 0.8700 | 0.0100 | 0.8650 | 0.0800 | 6164636.34 | 298121.87 | 12.8501 | high activity and material one-day price move |
| information_flow_watch | HSBC Championships: Katie Boulter vs Leylah Fernandez | 0.3000 | 0.3100 | 0.0100 | 0.3050 | -0.1700 | 402858.95 | 45854.05 | 12.7281 | high activity and material one-day price move |
| information_flow_watch | Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | 0.1360 | 0.1410 | 0.0050 | 0.1385 | -0.0585 | 5830535.78 | 240866.91 | 12.3527 | high activity and material one-day price move |
| information_flow_watch | Indiana Fever vs. Washington Mystics | 0.8000 | 0.8190 | 0.0190 | 0.8095 | 0.1445 | 347220.27 | 30501.97 | 12.0395 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 15, 2026? | 0.0500 | 0.0600 | 0.0100 | 0.0550 | -0.0300 | 2379061.53 | 636429.28 | 11.6340 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.1200 | 0.1300 | 0.0100 | 0.1250 | -0.0600 | 1439450.75 | 63895.21 | 11.3806 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.2300 | 0.2500 | 0.0200 | 0.2400 | -0.0650 | 459853.49 | 57869.85 | 10.8471 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by end of June? | 0.0900 | 0.1000 | 0.0100 | 0.0950 | -0.0300 | 611198.89 | 458825.78 | 10.8182 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.1900 | 0.2000 | 0.0100 | 0.1950 | -0.0500 | 316177.17 | 135808.60 | 10.6112 | high activity and material one-day price move |
| information_flow_watch | Will Vitality win IEM Cologne Major 2026? | 0.4400 | 0.4500 | 0.0100 | 0.4450 | -0.0200 | 698663.96 | 345365.60 | 10.4863 | high activity and material one-day price move |
| information_flow_watch | Will the U.S. invade Iran before 2027? | 0.1800 | 0.1900 | 0.0100 | 0.1850 | 0.0200 | 363775.41 | 448190.33 | 10.2931 | high activity and material one-day price move |
| information_flow_watch | Spread: Knicks (-1.5) | 0.4900 | 0.5000 | 0.0100 | 0.4950 | -0.0300 | 253490.41 | 489095.24 | 10.1965 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 9? | 0.0280 | 0.0290 | 0.0010 | 0.0285 | -0.0290 | 352040.58 | 65410.26 | 10.0132 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
