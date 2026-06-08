# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | Iran closes its airspace by June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.9825 | 8302263.46 | 3280649.58 | 31.5592 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 15? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.9245 | 1731413.70 | 1863208.70 | 29.4553 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 30? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.8160 | 1048651.31 | 1858380.81 | 27.0785 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.2600 | 0.2900 | 0.0300 | 0.2750 | 0.2200 | 2653631.59 | 41122.03 | 14.7239 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.3300 | 0.3600 | 0.0300 | 0.3450 | 0.2300 | 822082.99 | 23439.51 | 14.1897 | high activity and material one-day price move |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.7800 | 0.7900 | 0.0100 | 0.7850 | 0.1200 | 3239338.19 | 105259.79 | 13.0818 | high activity and material one-day price move |
| information_flow_watch | Will Microstrategy announce a Bitcoin purchase June 2-8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.1145 | 1120393.15 | 773798.32 | 12.8168 | high activity and material one-day price move |
| information_flow_watch | Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | 0.2280 | 0.2300 | 0.0020 | 0.2290 | -0.1025 | 2950024.09 | 75727.59 | 12.6267 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.2200 | 0.2500 | 0.0300 | 0.2350 | 0.1350 | 268864.69 | 57515.27 | 11.9036 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $58,000 on June 8? | 0.9960 | 0.9970 | 0.0010 | 0.9965 | 0.0715 | 362981.54 | 56559.86 | 10.7755 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 9? | 0.1000 | 0.1160 | 0.0160 | 0.1080 | 0.0745 | 287382.02 | 45185.05 | 10.7142 | high activity and material one-day price move |
| information_flow_watch | Bab el-Mandeb Strait effectively closed by June 30? | 0.0820 | 0.1230 | 0.0410 | 0.1025 | 0.0540 | 235944.76 | 69628.99 | 10.2358 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.2100 | 0.2300 | 0.0200 | 0.2200 | -0.0250 | 427967.29 | 103402.26 | 10.1836 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
