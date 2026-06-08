# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | 0.1900 | 0.2000 | 0.0100 | 135010.26 | 130713.11 | 347030.15 | 338432.55 | 338.3325 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | 0.8000 | 0.8100 | 0.0100 | 130713.11 | 135010.26 | 338432.55 | 347030.15 | 338.3325 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 108904.31 | 24623.25 | 594445.09 | 241801.22 | 241.7012 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 24623.25 | 108904.31 | 241801.22 | 616502.66 | 241.7012 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 8476.97 | 66104.32 | 587504.11 | 240477.30 | 240.3773 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 66104.32 | 8476.97 | 240477.30 | 587504.11 | 240.3773 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 21072.62 | 17026.07 | 167231.91 | 415515.62 | 167.1319 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 17026.07 | 21072.62 | 409661.46 | 154024.90 | 153.9249 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 1635.56 | 29127.37 | 131022.80 | 134331.92 | 130.9228 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 29127.37 | 1635.56 | 126428.35 | 121494.21 | 121.3942 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2700 | 0.2800 | 0.0100 | 8064.29 | 1281.99 | 72271.01 | 71876.95 | 71.7769 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7200 | 0.7300 | 0.0100 | 1281.99 | 8064.29 | 71876.95 | 81001.00 | 71.7769 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8200 | 0.8300 | 0.0100 | 220.95 | 2430.83 | 57707.22 | 50475.33 | 50.3753 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1700 | 0.1800 | 0.0100 | 2430.83 | 220.95 | 46867.36 | 47884.39 | 46.7674 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.1600 | 0.1700 | 0.0100 | 9221.60 | 8924.01 | 43859.26 | 38964.60 | 38.8646 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8300 | 0.8400 | 0.0100 | 8924.01 | 9221.60 | 38964.60 | 43859.26 | 38.8646 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | Yes | 0.5100 | 0.5200 | 0.0100 | 9234.82 | 125.00 | 70494.15 | 28363.85 | 28.2638 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | No | 0.4800 | 0.4900 | 0.0100 | 125.00 | 9234.82 | 28363.85 | 70494.15 | 28.2638 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.5200 | 0.5400 | 0.0200 | 269.51 | 557.88 | 83176.92 | 16768.16 | 16.5682 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.4600 | 0.4800 | 0.0200 | 557.88 | 269.51 | 16562.42 | 83676.92 | 16.3624 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
