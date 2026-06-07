# Prediction Markets Results

Data:

- source: Polymarket public Gamma API
- request: active and open markets ordered by 24h volume
- output: public microstructure screen

Polymarket documents Gamma market data as public market discovery data. This
screen does not use authenticated trading endpoints.

## Current Polymarket Microstructure Screen

Run:

```bash
uv run python -m strategies.prediction_markets.current_polymarket_microstructure --limit 200
```

This is not a trade instruction. It ranks markets where event modeling,
information-flow research, or market-making research might be worth building.

Top current rows:

| action | question | spread | 1d change | vol24h | liquidity | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| information_flow_watch | HSBC Championships, Qualification: Storm Hunter vs Aliaksandra Sasnovich | 0.0100 | 0.6750 | 238892.48 | 62744.53 | 22.6138 | high activity and material one-day price move |
| information_flow_watch | LoL: KT Rolster vs Dplus KIA (BO5) - LCK Road to MSI | 0.0010 | 0.4845 | 3734690.69 | 238686.06 | 20.5935 | high activity and material one-day price move |
| information_flow_watch | Will Kimi Antonelli win the 2026 F1 Monaco Grand Prix? | 0.0400 | 0.5700 | 241579.38 | 20713.71 | 20.2725 | high activity and material one-day price move |
| information_flow_watch | Bitcoin Up or Down on June 7? | 0.0010 | 0.4945 | 285041.08 | 66470.11 | 19.1191 | high activity and material one-day price move |
| information_flow_watch | Counter-Strike: Virtus.pro vs GenOne (BO3) - European Pro League Series 7 Group D | 0.0100 | 0.3150 | 216577.21 | 85705.90 | 15.4266 | high activity and material one-day price move |
| information_flow_watch | US x Iran permanent peace deal by June 30, 2026? | 0.0100 | -0.0600 | 662249.52 | 423023.96 | 11.4400 | high activity and material one-day price move |
| market_making_watch | Dota 2: Team Yandex vs LGD Gaming - Game 1 Winner | 0.0400 | 0.0000 | 1259623.34 | 36602.44 | 11.2821 | high activity with non-trivial visible spread |

Interpretation:

- This is a new, non-crypto-perp lane.
- The screen found active event markets with current information movement and
  tradable order books.
- It does not decide true probability. The next useful work is to attach
  external event models, CLOB depth, and adverse-selection checks.
