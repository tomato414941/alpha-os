# Current Polymarket Microstructure Screen

This screen looks for active event markets with enough public activity to justify prediction-model or market-making work. It is not a trade instruction.

| action | question | bid | ask | spread | mid | 1d change | vol24h | liq | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | HSBC Championships: Harriet Dart vs Liudmila Samsonova | 0.8800 | 0.8900 | 0.0100 | 0.8850 | 0.5900 | 495366.46 | 85111.15 | 21.3737 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 8? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.3930 | 9046361.17 | 3847351.04 | 19.8581 | high activity and material one-day price move |
| information_flow_watch | HSBC Championships: Karolina Pliskova vs McCartney Kessler | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.4845 | 571424.73 | 245375.25 | 19.5801 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 15? | 0.1600 | 0.1700 | 0.0100 | 0.1650 | -0.4100 | 1987779.61 | 49077.63 | 18.4605 | high activity and material one-day price move |
| information_flow_watch | Will Keiko Fujimori win the 2026 Peruvian presidential election? | 0.8400 | 0.8500 | 0.0100 | 0.8450 | 0.3300 | 4042690.07 | 147953.36 | 17.4751 | high activity and material one-day price move |
| information_flow_watch | Israel closes its airspace by June 30? | 0.2200 | 0.2400 | 0.0200 | 0.2300 | -0.3900 | 541903.61 | 36995.70 | 17.3193 | high activity and material one-day price move |
| information_flow_watch | Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | 0.1500 | 0.1610 | 0.0110 | 0.1555 | -0.3125 | 4415948.01 | 208617.24 | 17.2470 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 15? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.2945 | 2148045.52 | 1423405.74 | 16.9143 | high activity and material one-day price move |
| information_flow_watch | Iran closes its airspace by June 30? | 0.9990 | 1.0000 | 0.0010 | 0.9995 | 0.2620 | 1031332.34 | 1514616.71 | 15.9517 | high activity and material one-day price move |
| information_flow_watch | HSBC Championships: Katie Boulter vs Leylah Fernandez | 0.2700 | 0.2800 | 0.0100 | 0.2750 | -0.1850 | 320825.02 | 94128.76 | 13.0602 | high activity and material one-day price move |
| information_flow_watch | Will Netherlands win on 2026-06-08? | 0.9000 | 0.9100 | 0.0100 | 0.9050 | 0.1300 | 312450.78 | 44097.97 | 11.7906 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 30, 2026? | 0.1600 | 0.1700 | 0.0100 | 0.1650 | 0.0400 | 1124182.56 | 460582.84 | 11.3096 | high activity and material one-day price move |
| information_flow_watch | France vs. Northern Ireland: O/U 1.5 | 0.7500 | 0.7600 | 0.0100 | 0.7550 | -0.1000 | 270560.86 | 21855.38 | 10.9502 | high activity and material one-day price move |
| information_flow_watch | Boston Red Sox vs. New York Yankees | 0.4600 | 0.4900 | 0.0300 | 0.4750 | 0.0600 | 221838.50 | 154193.13 | 10.6016 | high activity and material one-day price move |
| information_flow_watch | Kharg Island no longer under Iranian control by June 30? | 0.0230 | 0.0240 | 0.0010 | 0.0235 | -0.0220 | 645642.19 | 224158.28 | 10.5088 | high activity and material one-day price move |
| information_flow_watch | US announces new Iran agreement/ceasefire extension by June 12? | 0.1300 | 0.1500 | 0.0200 | 0.1400 | 0.0550 | 284258.50 | 67079.06 | 10.3820 | high activity and material one-day price move |
| information_flow_watch | US-Iran nuclear deal by June 30? | 0.2000 | 0.2100 | 0.0100 | 0.2050 | 0.0200 | 344001.84 | 124145.96 | 10.0284 | high activity and material one-day price move |
| information_flow_watch | Bab el-Mandeb Strait effectively closed by June 30? | 0.0890 | 0.1170 | 0.0280 | 0.1030 | -0.0335 | 243228.83 | 75344.10 | 9.8603 | high activity and material one-day price move |

## Interpretation

`information_flow_watch` means a high-volume market moved materially in the last day with a tradable order book. `market_making_watch` means volume exists but the visible spread is still wide enough to deserve fill/adverse-selection research. This screen does not estimate true event probability.
