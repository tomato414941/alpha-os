# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1600 | 0.1700 | 0.0100 | 13709.73 | 17914.51 | 829677.32 | 117880.53 | 117.7805 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8300 | 0.8400 | 0.0100 | 17914.51 | 13709.73 | 117880.53 | 829677.32 | 117.7805 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | 0.2700 | 0.2800 | 0.0100 | 29527.05 | 402.33 | 115780.51 | 73419.59 | 73.3196 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | No | 0.7200 | 0.7300 | 0.0100 | 402.33 | 29527.05 | 73419.59 | 116268.51 | 73.3196 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.2000 | 0.2100 | 0.0100 | 9032.30 | 5876.70 | 55356.61 | 43946.22 | 43.8462 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.7900 | 0.8000 | 0.0100 | 5876.70 | 9032.30 | 43946.22 | 63422.72 | 43.8462 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.8300 | 0.8400 | 0.0100 | 1979.78 | 5275.05 | 74363.52 | 28288.35 | 28.1883 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.1600 | 0.1700 | 0.0100 | 5275.05 | 1979.78 | 28288.35 | 74363.52 | 28.1883 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.1500 | 0.1600 | 0.0100 | 293.29 | 5.00 | 79042.98 | 24058.15 | 23.9581 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.8400 | 0.8500 | 0.0100 | 5.00 | 293.29 | 24058.15 | 79042.98 | 23.9581 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | Yes | 0.0990 | 0.1170 | 0.0180 | 40.14 | 14.80 | 23516.54 | 13143.34 | 12.9633 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | No | 0.8830 | 0.9010 | 0.0180 | 14.80 | 40.14 | 13143.34 | 23516.54 | 12.9633 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.1580 | 0.1590 | 0.0010 | 1337.00 | 6167.26 | 12351.28 | 92108.05 | 12.3413 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.8410 | 0.8420 | 0.0010 | 6167.26 | 1337.00 | 92108.05 | 12351.28 | 12.3413 | visible depth exists near both sides |
| France vs. Northern Ireland: O/U 1.5 | No | 0.1100 | 0.1200 | 0.0100 | 5343.72 | 1116.61 | 21299.11 | 7465.62 | 7.3656 | visible depth exists near both sides |
| France vs. Northern Ireland: O/U 1.5 | Yes | 0.8800 | 0.8900 | 0.0100 | 1125.61 | 5343.72 | 7447.62 | 21299.11 | 7.3476 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8800 | 0.9100 | 0.0300 | 30.00 | 4385.68 | 7515.88 | 69645.47 | 7.2159 | visible depth exists near both sides |
| Will Netherlands win on 2026-06-08? | Yes | 0.9000 | 0.9100 | 0.0100 | 1836.07 | 5940.91 | 6679.96 | 24877.09 | 6.5800 | visible depth exists near both sides |
| Will Netherlands win on 2026-06-08? | No | 0.0900 | 0.1000 | 0.0100 | 5940.91 | 1836.07 | 24877.09 | 6679.96 | 6.5800 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.0900 | 0.1200 | 0.0300 | 4385.68 | 30.00 | 69645.47 | 5780.36 | 5.4804 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
