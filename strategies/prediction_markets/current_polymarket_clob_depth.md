# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | 0.1900 | 0.2000 | 0.0100 | 128045.87 | 195352.61 | 318278.21 | 438456.62 | 318.1782 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | 0.8000 | 0.8100 | 0.0100 | 195352.61 | 128045.87 | 438456.62 | 318278.21 | 318.1782 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 14882.98 | 66090.74 | 592335.54 | 241521.69 | 241.4217 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 66090.74 | 14882.98 | 241521.69 | 592335.54 | 241.4217 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 23487.09 | 5502.29 | 169234.32 | 418636.13 | 169.1343 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 5502.29 | 23487.09 | 413605.97 | 156027.31 | 155.9273 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 373.00 | 20795.12 | 126259.53 | 131734.80 | 126.1595 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 20795.12 | 373.00 | 123831.23 | 119442.12 | 119.3421 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7100 | 0.7200 | 0.0100 | 11288.94 | 3790.26 | 76195.14 | 89013.04 | 76.0951 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2800 | 0.2900 | 0.0100 | 3790.26 | 11288.94 | 87866.46 | 71968.96 | 71.8690 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8200 | 0.8300 | 0.0100 | 1755.69 | 3226.35 | 59845.75 | 55289.72 | 55.1897 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1700 | 0.1800 | 0.0100 | 3226.35 | 1755.69 | 51694.75 | 50000.02 | 49.9000 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | Yes | 0.3600 | 0.4200 | 0.0600 | 79.37 | 19.40 | 51819.22 | 26156.22 | 25.5562 | spread is wide despite visible depth |
| Boston Red Sox vs. New York Yankees | No | 0.5800 | 0.6400 | 0.0600 | 19.40 | 79.37 | 26156.22 | 51819.22 | 25.5562 | spread is wide despite visible depth |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.1500 | 0.1600 | 0.0100 | 10277.11 | 850.87 | 21718.66 | 76853.55 | 21.6187 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8400 | 0.8500 | 0.0100 | 850.87 | 10277.11 | 76853.55 | 21718.66 | 21.6187 | visible depth exists near both sides |
| Iran leadership change by June 30? | Yes | 0.0670 | 0.0680 | 0.0010 | 443.50 | 260.78 | 118764.90 | 13690.58 | 13.6806 | visible depth exists near both sides |
| Iran leadership change by June 30? | No | 0.9320 | 0.9330 | 0.0010 | 260.78 | 443.50 | 13690.58 | 118764.90 | 13.6806 | visible depth exists near both sides |
| Israel closes its airspace by June 30? | Yes | 0.6000 | 0.6300 | 0.0300 | 121.18 | 8.41 | 45213.29 | 5739.99 | 5.4400 | visible depth exists near both sides |
| Israel closes its airspace by June 30? | No | 0.3700 | 0.4000 | 0.0300 | 8.41 | 121.18 | 5739.99 | 45213.29 | 5.4400 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
