# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 9119.39 | 109778.61 | 633081.23 | 287971.81 | 287.8718 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 109778.61 | 9119.39 | 287971.81 | 633081.23 | 287.8718 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 75435.71 | 25766.20 | 956908.46 | 241668.76 | 241.5688 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 25766.20 | 75435.71 | 241668.76 | 978966.03 | 241.5688 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 15328.21 | 11975.88 | 159952.74 | 409925.43 | 159.8527 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 11975.88 | 15328.21 | 403104.43 | 146745.73 | 146.6457 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 4267.54 | 18342.05 | 120719.88 | 119222.26 | 119.1223 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 18342.05 | 4267.54 | 111318.69 | 111770.75 | 111.2187 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7200 | 0.7300 | 0.0100 | 29.00 | 7790.26 | 71701.32 | 74092.10 | 71.6013 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2700 | 0.2800 | 0.0100 | 7790.26 | 29.00 | 65361.91 | 71701.32 | 65.2619 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1800 | 0.1900 | 0.0100 | 272.18 | 2049.08 | 47600.67 | 53889.18 | 47.5007 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8100 | 0.8200 | 0.0100 | 2049.08 | 272.18 | 53889.18 | 47600.67 | 47.5007 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | Yes | 0.5200 | 0.5300 | 0.0100 | 2838.45 | 9063.61 | 55441.72 | 44723.75 | 44.6238 | visible depth exists near both sides |
| Libema Open: Otto Virtanen vs Kamil Majchrzak | No | 0.4700 | 0.4800 | 0.0100 | 9063.61 | 2838.45 | 44723.75 | 55660.72 | 44.6238 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.1600 | 0.1800 | 0.0200 | 1581.41 | 6563.24 | 26227.86 | 22416.50 | 22.2165 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8200 | 0.8400 | 0.0200 | 6563.24 | 1581.41 | 22416.50 | 26227.86 | 22.2165 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.7700 | 0.7900 | 0.0200 | 2829.90 | 634.30 | 23663.67 | 18648.95 | 18.4489 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.2100 | 0.2300 | 0.0200 | 634.30 | 2829.90 | 18648.95 | 23663.67 | 18.4489 | visible depth exists near both sides |
| Iran leadership change by June 30? | Yes | 0.0660 | 0.0690 | 0.0030 | 1265.00 | 10.99 | 91129.28 | 15192.37 | 15.1624 | visible depth exists near both sides |
| Iran leadership change by June 30? | No | 0.9310 | 0.9340 | 0.0030 | 10.99 | 1265.00 | 15192.37 | 91129.28 | 15.1624 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
