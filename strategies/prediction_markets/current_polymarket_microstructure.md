# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | Iran closes its airspace by June 8? | 0.9980 | 0.9990 | 0.0010 | 0.9985 | 0.9810 | 7288496.53 | 3189299.47 | 31.4527 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 15? | 0.9980 | 0.9990 | 0.0010 | 0.9985 | 0.9205 | 1565905.91 | 1373098.82 | 29.2570 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 30? | 0.9980 | 0.9990 | 0.0010 | 0.9985 | 0.8115 | 1010882.69 | 1509786.67 | 26.9249 | high activity and material one-day price move |
| information_flow_watch | Will Bahrain vs. Syria end in a draw? | 0.9710 | 0.9760 | 0.0050 | 0.9735 | 0.6635 | 240443.85 | 53607.49 | 22.3559 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.6200 | 0.6500 | 0.0300 | 0.6350 | 0.5200 | 677805.33 | 59707.74 | 20.0957 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.4700 | 0.4800 | 0.0100 | 0.4750 | 0.4100 | 2253444.89 | 87229.09 | 18.6221 | high activity and material one-day price move |
| information_flow_watch | Will Bahrain win on 2026-06-09? | 0.0070 | 0.0140 | 0.0070 | 0.0105 | -0.3695 | 261744.04 | 56983.88 | 16.5332 | high activity and material one-day price move |
| information_flow_watch | Will Microstrategy announce a Bitcoin purchase June 2-8? | 0.9710 | 0.9750 | 0.0040 | 0.9730 | 0.1180 | 780943.36 | 173601.21 | 12.3535 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 15, 2026? | 0.0400 | 0.0500 | 0.0100 | 0.0450 | -0.0200 | 2638620.77 | 1304538.12 | 11.6294 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by end of June? | 0.0900 | 0.1000 | 0.0100 | 0.0950 | -0.0400 | 774195.02 | 538377.51 | 11.1665 | high activity and material one-day price move |
| information_flow_watch | LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | 0.1900 | 0.2000 | 0.0100 | 0.1950 | -0.0600 | 397512.11 | 507674.10 | 11.0487 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 30, 2026? | 0.1400 | 0.1500 | 0.0100 | 0.1450 | -0.0200 | 1171147.26 | 460866.33 | 10.9287 | high activity and material one-day price move |
| market_making_watch | Boston Red Sox vs. New York Yankees | 0.3700 | 0.4200 | 0.0500 | 0.3950 | -0.0100 | 216934.44 | 156955.82 | 10.9077 | high activity with non-trivial visible spread |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.1800 | 0.1900 | 0.0100 | 0.1850 | -0.0600 | 350776.16 | 131786.84 | 10.8559 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $58,000 on June 8? | 0.9970 | 0.9980 | 0.0010 | 0.9975 | 0.0825 | 211424.38 | 52403.38 | 10.6941 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.1600 | 0.1700 | 0.0100 | 0.1650 | 0.0700 | 187992.79 | 87459.92 | 10.5369 | high activity and material one-day price move |
| information_flow_watch | Will the U.S. invade Iran before 2027? | 0.1700 | 0.1800 | 0.0100 | 0.1750 | 0.0200 | 362006.27 | 455562.66 | 10.2969 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by July 31, 2026? | 0.2800 | 0.2900 | 0.0100 | 0.2850 | -0.0300 | 228031.91 | 314404.86 | 10.2601 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by July 31? | 0.2800 | 0.2900 | 0.0100 | 0.2850 | -0.0300 | 185294.80 | 199186.79 | 9.9911 | high activity and material one-day price move |
| information_flow_watch | Iran leadership change by June 30? | 0.0670 | 0.0680 | 0.0010 | 0.0675 | 0.0275 | 313341.75 | 55801.45 | 9.9277 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
