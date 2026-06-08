# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | Iran closes its airspace by June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.9270 | 8491229.02 | 2942372.56 | 30.4387 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 15? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.8770 | 1786566.00 | 1830639.11 | 28.5167 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 30? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.7940 | 1036154.36 | 1844972.50 | 26.6320 | high activity and material one-day price move |
| information_flow_watch | Libema Open: Daria Snigur vs Paula Badosa | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.5345 | 567188.58 | 225006.84 | 20.5588 | high activity and material one-day price move |
| information_flow_watch | Libema Open: Marin Cilic vs Denis Shapovalov | 0.1900 | 0.2000 | 0.0100 | 0.1950 | -0.3400 | 376714.53 | 53170.89 | 16.1229 | high activity and material one-day price move |
| information_flow_watch | Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | 0.5000 | 0.5100 | 0.0100 | 0.5050 | -0.1600 | 445084.52 | 156386.09 | 12.8481 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.2900 | 0.3100 | 0.0200 | 0.3000 | 0.1200 | 720290.61 | 36916.29 | 12.0408 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.2200 | 0.2300 | 0.0100 | 0.2250 | 0.1300 | 270824.05 | 59574.69 | 11.8380 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. New York Yankees | 0.5000 | 0.5200 | 0.0200 | 0.5100 | 0.1150 | 262931.85 | 135459.25 | 11.7571 | high activity and material one-day price move |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.7000 | 0.7100 | 0.0100 | 0.7050 | 0.0400 | 3462935.98 | 123067.25 | 11.5526 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $58,000 on June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.0945 | 393920.17 | 124541.38 | 11.4509 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.1800 | 0.2000 | 0.0200 | 0.1900 | 0.0500 | 2589812.47 | 45427.17 | 11.3471 | high activity and material one-day price move |
| information_flow_watch | Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | 0.3080 | 0.3100 | 0.0020 | 0.3090 | -0.0210 | 3260617.10 | 117770.81 | 11.1464 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 30, 2026? | 0.1700 | 0.1800 | 0.0100 | 0.1750 | 0.0300 | 1213217.24 | 455232.70 | 11.1451 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by end of June? | 0.1000 | 0.1100 | 0.0100 | 0.1050 | -0.0200 | 975343.69 | 509944.63 | 10.8531 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.1900 | 0.2000 | 0.0100 | 0.1950 | -0.0500 | 452297.44 | 129417.01 | 10.7601 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 9? | 0.0810 | 0.1030 | 0.0220 | 0.0920 | 0.0600 | 301687.81 | 63017.99 | 10.5160 | high activity and material one-day price move |
| information_flow_watch | Will the U.S. invade Iran before 2027? | 0.1700 | 0.1800 | 0.0100 | 0.1750 | 0.0200 | 469895.96 | 464000.66 | 10.4129 | high activity and material one-day price move |
| information_flow_watch | Bab el-Mandeb Strait effectively closed by June 30? | 0.0960 | 0.1160 | 0.0200 | 0.1060 | 0.0480 | 248731.03 | 77017.82 | 10.1715 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $66,000 on June 8? | 0.0010 | 0.0020 | 0.0010 | 0.0015 | -0.0465 | 322297.03 | 32505.67 | 10.0900 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $56,000 on June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.0200 | 363965.63 | 127844.35 | 9.9215 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
