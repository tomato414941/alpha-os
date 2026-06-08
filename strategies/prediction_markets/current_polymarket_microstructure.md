# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | Iran closes its airspace by June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.9755 | 8286239.37 | 3261728.77 | 31.4171 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 15? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.9175 | 1761238.92 | 1834671.07 | 29.3207 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 30? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.8140 | 1044037.02 | 1858126.67 | 27.0366 | high activity and material one-day price move |
| information_flow_watch | Game Handicap: BLG (-2.5) vs Anyone's Legend (+2.5) | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.6795 | 311090.58 | 219541.06 | 23.1259 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.3400 | 0.3600 | 0.0200 | 0.3500 | 0.2350 | 805317.18 | 37600.04 | 14.3942 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.2300 | 0.2400 | 0.0100 | 0.2350 | 0.1700 | 2627847.48 | 48395.54 | 13.7757 | high activity and material one-day price move |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.7800 | 0.7900 | 0.0100 | 0.7850 | 0.1200 | 3257111.16 | 105453.83 | 13.0850 | high activity and material one-day price move |
| information_flow_watch | Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | 0.2110 | 0.2180 | 0.0070 | 0.2145 | -0.1170 | 3005828.86 | 76273.37 | 12.9227 | high activity and material one-day price move |
| information_flow_watch | Will Microstrategy announce a Bitcoin purchase June 2-8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.1095 | 1115653.14 | 762622.59 | 12.7120 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.2200 | 0.2300 | 0.0100 | 0.2250 | 0.1250 | 269961.30 | 60087.40 | 11.7351 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by end of June? | 0.1000 | 0.1100 | 0.0100 | 0.1050 | -0.0200 | 981151.08 | 476869.55 | 10.8498 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $58,000 on June 8? | 0.9970 | 0.9990 | 0.0020 | 0.9980 | 0.0730 | 369599.65 | 59230.84 | 10.8244 | high activity and material one-day price move |
| market_making_watch | Israel closes its airspace by June 8? | 0.0680 | 0.0900 | 0.0220 | 0.0790 | 0.0000 | 699045.97 | 35203.93 | 10.6484 | high activity with non-trivial visible spread |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 9? | 0.0860 | 0.0930 | 0.0070 | 0.0895 | 0.0560 | 293953.80 | 64367.87 | 10.4409 | high activity and material one-day price move |
| information_flow_watch | Will the U.S. invade Iran before 2027? | 0.1700 | 0.1800 | 0.0100 | 0.1750 | 0.0200 | 428163.24 | 451148.34 | 10.3728 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.2100 | 0.2200 | 0.0100 | 0.2150 | -0.0300 | 433697.91 | 121188.91 | 10.3342 | high activity and material one-day price move |
| information_flow_watch | Bab el-Mandeb Strait effectively closed by June 30? | 0.0960 | 0.1160 | 0.0200 | 0.1060 | 0.0565 | 238106.88 | 68574.50 | 10.3080 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. New York Yankees | 0.4000 | 0.4200 | 0.0200 | 0.4100 | -0.0400 | 267257.64 | 135251.00 | 10.2638 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $56,000 on June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.0200 | 359158.74 | 97975.19 | 9.8559 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
