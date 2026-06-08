# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | Iran closes its airspace by June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.9315 | 8500332.98 | 3634276.36 | 30.5752 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 15? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.8495 | 1833987.10 | 2178434.98 | 28.0180 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 30? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.7875 | 1033254.37 | 1786429.98 | 26.4937 | high activity and material one-day price move |
| information_flow_watch | Libema Open: Daria Snigur vs Paula Badosa | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.5345 | 567747.66 | 280392.32 | 20.6071 | high activity and material one-day price move |
| information_flow_watch | Libema Open: Mia Pohankova vs Clara Tauson | 0.6900 | 0.7000 | 0.0100 | 0.6950 | 0.3900 | 275795.14 | 137738.08 | 17.1625 | high activity and material one-day price move |
| information_flow_watch | Stuttgart Open: Tommy Paul vs Giovanni Mpetshi Perricard | 0.5000 | 0.5100 | 0.0100 | 0.5050 | -0.1600 | 445238.44 | 155371.58 | 12.8468 | high activity and material one-day price move |
| information_flow_watch | Libema Open: Marin Cilic vs Denis Shapovalov | 0.3800 | 0.3900 | 0.0100 | 0.3850 | -0.1500 | 409222.74 | 123705.52 | 12.5512 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.2200 | 0.2300 | 0.0100 | 0.2250 | 0.1350 | 269737.46 | 59371.81 | 11.9355 | high activity and material one-day price move |
| market_making_watch | Israel closes its airspace by June 15? | 0.1500 | 0.1800 | 0.0300 | 0.1650 | 0.0100 | 2605422.77 | 39713.75 | 11.5404 | high activity with non-trivial visible spread |
| information_flow_watch | Will the price of Bitcoin be above $58,000 on June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.0945 | 393913.80 | 121312.94 | 11.4452 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.2600 | 0.3000 | 0.0400 | 0.2800 | 0.0900 | 720493.40 | 39604.90 | 11.4366 | high activity and material one-day price move |
| market_making_watch | Boston Red Sox vs. New York Yankees | 0.4500 | 0.5200 | 0.0700 | 0.4850 | 0.0650 | 262812.52 | 135538.12 | 11.1768 | high activity with non-trivial visible spread |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.6800 | 0.6900 | 0.0100 | 0.6850 | 0.0200 | 3472578.19 | 122027.84 | 11.1521 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 30, 2026? | 0.1700 | 0.1800 | 0.0100 | 0.1750 | 0.0300 | 1217226.29 | 453941.75 | 11.1460 | high activity and material one-day price move |
| information_flow_watch | Strait of Hormuz traffic returns to normal by end of June? | 0.1000 | 0.1100 | 0.0100 | 0.1050 | -0.0200 | 882647.68 | 522583.17 | 10.8144 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.1900 | 0.2000 | 0.0100 | 0.1950 | -0.0500 | 452667.90 | 134240.26 | 10.7682 | high activity and material one-day price move |
| information_flow_watch | Will the U.S. invade Iran before 2027? | 0.1700 | 0.1800 | 0.0100 | 0.1750 | 0.0200 | 474893.65 | 434226.80 | 10.4035 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 9? | 0.0770 | 0.0880 | 0.0110 | 0.0825 | 0.0510 | 301828.31 | 61096.26 | 10.3405 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $66,000 on June 8? | 0.0010 | 0.0020 | 0.0010 | 0.0015 | -0.0420 | 343112.01 | 25419.12 | 9.9796 | high activity and material one-day price move |
| information_flow_watch | Will the price of Bitcoin be above $56,000 on June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.0200 | 363959.00 | 123296.45 | 9.9137 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
