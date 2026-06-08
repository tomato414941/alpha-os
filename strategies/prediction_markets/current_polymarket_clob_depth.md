# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 67357.24 | 42203.85 | 391099.60 | 206339.84 | 206.2398 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 42203.85 | 67357.24 | 206339.84 | 391099.60 | 206.2398 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | 0.8000 | 0.8100 | 0.0100 | 96543.03 | 16234.13 | 401754.14 | 127477.59 | 127.3776 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | 0.1900 | 0.2000 | 0.0100 | 16234.13 | 96543.03 | 123477.59 | 397754.14 | 123.3776 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 44900.89 | 9356.63 | 241875.40 | 113530.99 | 113.4310 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 9356.63 | 44900.89 | 113530.99 | 281869.97 | 113.4310 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 11938.27 | 5690.81 | 103340.04 | 350984.64 | 103.2400 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.3000 | 0.3100 | 0.0100 | 24857.87 | 1476.19 | 98733.01 | 120124.13 | 98.6330 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.6900 | 0.7000 | 0.0100 | 1476.19 | 24857.87 | 120124.13 | 98733.01 | 98.6330 | visible depth exists near both sides |
| San Francisco Giants vs. Chicago Cubs | Yes | 0.4500 | 0.4600 | 0.0100 | 59094.07 | 260.16 | 96211.22 | 98221.82 | 96.1112 | visible depth exists near both sides |
| San Francisco Giants vs. Chicago Cubs | No | 0.5400 | 0.5500 | 0.0100 | 260.16 | 59094.07 | 98215.82 | 96211.22 | 96.1112 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 5690.81 | 11938.27 | 346500.92 | 90133.03 | 90.0330 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.7000 | 0.7060 | 0.0060 | 100.00 | 1431.47 | 38626.81 | 46713.65 | 38.5668 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.2940 | 0.3000 | 0.0060 | 1431.47 | 100.00 | 46713.65 | 38626.81 | 38.5668 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.7900 | 0.8000 | 0.0100 | 501.57 | 1458.29 | 32590.92 | 38738.20 | 32.4909 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7100 | 0.7300 | 0.0200 | 2927.87 | 6443.58 | 32330.42 | 86925.59 | 32.1304 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | Yes | 0.4100 | 0.4400 | 0.0300 | 257.53 | 1068.31 | 29988.09 | 30943.52 | 29.6881 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | No | 0.5600 | 0.5900 | 0.0300 | 1068.31 | 257.53 | 30943.52 | 29988.09 | 29.6881 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.2000 | 0.2100 | 0.0100 | 1458.29 | 501.57 | 26736.42 | 32590.92 | 26.6364 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2700 | 0.2900 | 0.0200 | 6443.58 | 2927.87 | 72069.04 | 20395.65 | 20.1957 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
