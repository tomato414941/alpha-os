# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | Yes | 0.1900 | 0.2000 | 0.0100 | 128015.94 | 146312.36 | 314634.82 | 353698.80 | 314.5348 | visible depth exists near both sides |
| LoL: Anyone's Legend vs Bilibili Gaming (BO5) - LPL Playoffs | No | 0.8000 | 0.8100 | 0.0100 | 146312.36 | 128015.94 | 353698.80 | 314634.82 | 314.5348 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 14766.12 | 65915.57 | 591947.46 | 239801.14 | 239.7011 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 65915.57 | 14766.12 | 239801.14 | 591947.46 | 239.7011 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 129179.06 | 20696.09 | 619217.11 | 236052.42 | 235.9524 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 20696.09 | 129179.06 | 236052.42 | 640163.68 | 235.9524 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8200 | 0.8300 | 0.0100 | 22831.30 | 5245.35 | 165491.90 | 385880.53 | 165.3919 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1700 | 0.1800 | 0.0100 | 5245.35 | 22831.30 | 380846.37 | 152284.89 | 152.1849 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | No | 0.7100 | 0.7200 | 0.0100 | 373.00 | 20792.27 | 132371.91 | 132332.12 | 132.2321 | visible depth exists near both sides |
| US x Iran permanent peace deal by July 31, 2026? | Yes | 0.2800 | 0.2900 | 0.0100 | 20792.27 | 373.00 | 124428.55 | 123843.32 | 123.7433 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | No | 0.7100 | 0.7200 | 0.0100 | 11551.02 | 4286.02 | 74906.65 | 89133.55 | 74.8067 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by July 31? | Yes | 0.2800 | 0.2900 | 0.0100 | 4286.02 | 11551.02 | 87986.97 | 70680.47 | 70.5805 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8200 | 0.8300 | 0.0100 | 1685.75 | 3259.19 | 59984.82 | 47452.64 | 47.3526 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1700 | 0.1800 | 0.0100 | 3259.19 | 1685.75 | 44690.67 | 50268.14 | 44.5907 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | Yes | 0.1500 | 0.1600 | 0.0100 | 9415.34 | 2499.73 | 34379.88 | 32325.16 | 32.2252 | visible depth exists near both sides |
| US announces new Iran agreement/ceasefire extension by June 12? | No | 0.8400 | 0.8500 | 0.0100 | 2499.73 | 9415.34 | 32325.16 | 34379.88 | 32.2252 | visible depth exists near both sides |
| Iran leadership change by June 30? | Yes | 0.0670 | 0.0700 | 0.0030 | 443.50 | 736.18 | 45616.57 | 26668.88 | 26.6389 | visible depth exists near both sides |
| Iran leadership change by June 30? | No | 0.9300 | 0.9330 | 0.0030 | 736.18 | 443.50 | 26668.88 | 45616.57 | 26.6389 | visible depth exists near both sides |
| Boston Red Sox vs. New York Yankees | Yes | 0.3600 | 0.4300 | 0.0700 | 78.92 | 373.60 | 51818.77 | 26096.07 | 25.3961 | spread is wide despite visible depth |
| Boston Red Sox vs. New York Yankees | No | 0.5700 | 0.6400 | 0.0700 | 373.60 | 78.92 | 26096.07 | 51818.77 | 25.3961 | spread is wide despite visible depth |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
