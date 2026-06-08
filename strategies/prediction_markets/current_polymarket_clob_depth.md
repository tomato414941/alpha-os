# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | 0.0500 | 0.0600 | 0.0100 | 110303.92 | 1063.33 | 2225716.26 | 210807.01 | 210.7070 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | No | 0.9400 | 0.9500 | 0.0100 | 1063.33 | 110303.92 | 210807.01 | 2225716.26 | 210.7070 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.1000 | 0.1100 | 0.0100 | 47383.42 | 6717.20 | 276192.73 | 149950.72 | 149.8507 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.8900 | 0.9000 | 0.0100 | 6717.20 | 47383.42 | 149950.72 | 276192.73 | 149.8507 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | 0.1900 | 0.2000 | 0.0100 | 7876.74 | 8452.91 | 111248.29 | 288688.03 | 111.1483 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | 0.8000 | 0.8100 | 0.0100 | 8452.91 | 7876.74 | 288688.03 | 111248.29 | 111.1483 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1500 | 0.1600 | 0.0100 | 1135.08 | 11970.14 | 148711.05 | 109131.19 | 109.0312 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8400 | 0.8500 | 0.0100 | 11970.14 | 1135.08 | 109131.19 | 148711.05 | 109.0312 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.3000 | 0.3100 | 0.0100 | 24965.25 | 55.25 | 85918.32 | 98681.33 | 85.8183 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.6900 | 0.7000 | 0.0100 | 55.25 | 24965.25 | 98681.33 | 85918.32 | 85.8183 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | Yes | 0.4000 | 0.4400 | 0.0400 | 5781.46 | 1402.66 | 30114.02 | 31277.87 | 29.7140 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | No | 0.5600 | 0.6000 | 0.0400 | 1402.66 | 5781.46 | 31277.87 | 30114.02 | 29.7140 | visible depth exists near both sides |
| San Francisco Giants vs. Chicago Cubs | Yes | 0.4600 | 0.4700 | 0.0100 | 7147.82 | 4037.00 | 22421.27 | 28110.81 | 22.3213 | visible depth exists near both sides |
| San Francisco Giants vs. Chicago Cubs | No | 0.5300 | 0.5400 | 0.0100 | 4037.00 | 4447.82 | 28074.81 | 19757.27 | 19.6573 | visible depth exists near both sides |
| San Francisco Giants vs. Chicago Cubs: O/U 8.5 | Yes | 0.3100 | 0.3200 | 0.0100 | 15.00 | 99.00 | 14825.81 | 15868.34 | 14.7258 | visible depth exists near both sides |
| San Francisco Giants vs. Chicago Cubs: O/U 8.5 | No | 0.6800 | 0.6900 | 0.0100 | 99.00 | 15.00 | 15868.34 | 14825.81 | 14.7258 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.5500 | 0.5800 | 0.0300 | 30.00 | 9601.80 | 10633.16 | 14705.64 | 10.3332 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.4200 | 0.4500 | 0.0300 | 9601.80 | 30.00 | 14705.64 | 10633.16 | 10.3332 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.5500 | 0.5780 | 0.0280 | 2696.70 | 500.00 | 10549.44 | 8253.21 | 7.9732 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.4220 | 0.4500 | 0.0280 | 500.00 | 2696.70 | 8253.21 | 10536.44 | 7.9732 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
