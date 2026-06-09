# Current Polymarket CLOB Depth

This checks visible CLOB depth for unsettled current microstructure monitor markets first, then falls back to near-certain markets only if needed. It is not a trade instruction.

| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | 0.0500 | 0.0600 | 0.0100 | 72138.55 | 74391.54 | 2709859.05 | 290869.76 | 290.7698 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | No | 0.9400 | 0.9500 | 0.0100 | 74391.54 | 72138.55 | 290869.76 | 2709859.05 | 290.7698 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | Yes | 0.0900 | 0.1000 | 0.0100 | 127169.66 | 24000.40 | 704839.91 | 167837.82 | 167.7378 | visible depth exists near both sides |
| Strait of Hormuz traffic returns to normal by end of June? | No | 0.9000 | 0.9100 | 0.0100 | 24000.40 | 127169.66 | 167837.82 | 704839.91 | 167.7378 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | Yes | 0.8600 | 0.8700 | 0.0100 | 47189.19 | 5223.90 | 107154.99 | 177624.45 | 107.0550 | visible depth exists near both sides |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | No | 0.1300 | 0.1400 | 0.0100 | 5223.90 | 47189.19 | 177624.45 | 107154.99 | 107.0550 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | Yes | 0.1350 | 0.1420 | 0.0070 | 46.87 | 1327.34 | 144058.94 | 75369.03 | 75.2990 | visible depth exists near both sides |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | No | 0.8580 | 0.8650 | 0.0070 | 1292.12 | 46.87 | 75333.81 | 144058.94 | 75.2638 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | No | 0.8000 | 0.8200 | 0.0200 | 7654.08 | 5415.34 | 63977.46 | 90571.01 | 63.7775 | visible depth exists near both sides |
| US-Iran nuclear deal by June 30? | Yes | 0.1800 | 0.2000 | 0.0200 | 5415.34 | 7654.08 | 90571.01 | 63977.46 | 63.7775 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | Yes | 0.2900 | 0.3000 | 0.0100 | 8816.48 | 28.16 | 65768.49 | 24490.77 | 24.3908 | visible depth exists near both sides |
| HSBC Championships: Katie Boulter vs Leylah Fernandez | No | 0.7000 | 0.7100 | 0.0100 | 28.16 | 8753.01 | 24490.77 | 65705.02 | 24.3908 | visible depth exists near both sides |
| Philadelphia Phillies vs. Toronto Blue Jays | Yes | 0.9300 | 0.9400 | 0.0100 | 9104.25 | 6151.65 | 22392.00 | 35047.12 | 22.2920 | visible depth exists near both sides |
| Philadelphia Phillies vs. Toronto Blue Jays | No | 0.0600 | 0.0700 | 0.0100 | 6151.65 | 9104.25 | 35047.12 | 22392.00 | 22.2920 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | Yes | 0.1200 | 0.1300 | 0.0100 | 12324.27 | 7302.12 | 90992.57 | 21244.15 | 21.1441 | visible depth exists near both sides |
| Israel closes its airspace by June 15? | No | 0.8700 | 0.8800 | 0.0100 | 7302.12 | 12324.27 | 21244.15 | 90992.57 | 21.1441 | visible depth exists near both sides |
| Will Vitality win IEM Cologne Major 2026? | Yes | 0.4400 | 0.4700 | 0.0300 | 61298.10 | 3439.99 | 169002.13 | 19072.97 | 18.7730 | visible depth exists near both sides |
| Will Vitality win IEM Cologne Major 2026? | No | 0.5300 | 0.5600 | 0.0300 | 3439.99 | 61298.10 | 16604.97 | 169002.13 | 16.3050 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | Yes | 0.0980 | 0.1120 | 0.0140 | 5.81 | 72.62 | 25166.17 | 15407.50 | 15.2675 | visible depth exists near both sides |
| Bab el-Mandeb Strait effectively closed by June 30? | No | 0.8880 | 0.9020 | 0.0140 | 72.62 | 5.81 | 15407.50 | 25166.17 | 15.2675 | visible depth exists near both sides |

## Interpretation

Depth is measured in outcome-token size, not guaranteed executable USD. A high score means visible public depth exists near top of book; it does not prove queue priority, fill probability, or adverse-selection edge.
