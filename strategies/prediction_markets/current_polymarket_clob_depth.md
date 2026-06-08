# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | 0.0500 | 0.0600 | 0.0100 | 83672.20 | 54360.19 | 2727930.02 | 291851.95 | 291.7519 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | No | 0.9400 | 0.9500 | 0.0100 | 54360.19 | 83672.14 | 291851.95 | 2727929.96 | 291.7519 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.8800 | 0.8900 | 0.0100 | 8453.66 | 23512.71 | 196150.03 | 506130.27 | 196.0500 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.1100 | 0.1200 | 0.0100 | 23512.71 | 8453.66 | 506130.27 | 191412.03 | 191.3120 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 129061.58 | 4086.65 | 696625.61 | 168293.01 | 168.1930 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 4086.65 | 129061.58 | 168293.01 | 696625.61 | 168.1930 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.1240 | 0.1250 | 0.0010 | 85.31 | 9060.18 | 140083.43 | 111159.88 | 111.1499 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.8750 | 0.8760 | 0.0010 | 9060.18 | 85.31 | 111159.88 | 140083.43 | 111.1499 | visible depth exists near both sides |
| Philadelphia Phillies vs. Toronto Blue Jays | Yes | 0.8200 | 0.8300 | 0.0100 | 74447.72 | 57318.59 | 100088.97 | 133474.16 | 99.9890 | visible depth exists near both sides |
| Philadelphia Phillies vs. Toronto Blue Jays | No | 0.1700 | 0.1800 | 0.0100 | 57318.59 | 74447.72 | 131793.16 | 98890.48 | 98.7905 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | Yes | 0.1800 | 0.1900 | 0.0100 | 1326.15 | 12588.91 | 452694.55 | 96573.76 | 96.4738 | visible depth exists near both sides |
| Will the U.S. invade Iran before 2027? | No | 0.8100 | 0.8200 | 0.0100 | 12588.91 | 1326.15 | 96573.76 | 452694.55 | 96.4738 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1800 | 0.1900 | 0.0100 | 5845.27 | 1184.20 | 82072.43 | 65795.52 | 65.6955 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8100 | 0.8200 | 0.0100 | 1184.20 | 5845.27 | 65795.52 | 82072.43 | 65.6955 | visible depth exists near both sides |
| Will Vitality win IEM Cologne Major 2026? | Yes | 0.4400 | 0.4500 | 0.0100 | 1742.16 | 44364.98 | 42048.47 | 68722.64 | 41.9485 | visible depth exists near both sides |
| Will Vitality win IEM Cologne Major 2026? | No | 0.5500 | 0.5600 | 0.0100 | 44364.98 | 1742.16 | 68722.64 | 42048.47 | 41.9485 | visible depth exists near both sides |
| Indiana Fever vs. Washington Mystics | Yes | 0.7800 | 0.7900 | 0.0100 | 121.20 | 931.76 | 30531.26 | 28139.48 | 28.0395 | visible depth exists near both sides |
| Indiana Fever vs. Washington Mystics | No | 0.2100 | 0.2200 | 0.0100 | 931.76 | 121.20 | 28139.48 | 30531.26 | 28.0395 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | 0.3000 | 0.3100 | 0.0100 | 1824.10 | 6215.75 | 63968.85 | 27225.00 | 27.1250 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | No | 0.6900 | 0.7000 | 0.0100 | 6215.75 | 1824.10 | 27225.00 | 63968.85 | 27.1250 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
