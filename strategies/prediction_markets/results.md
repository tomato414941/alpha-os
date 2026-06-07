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
uv run python -m strategies.prediction_markets.current_polymarket_microstructure_monitor --samples 5 --delay-seconds 10 --limit 200
uv run python -m strategies.prediction_markets.current_polymarket_clob_depth --top-markets 10
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

## Current Polymarket Microstructure Monitor

This repeats the public microstructure screen over a short window.

| action | question | obs | mean score | min score | spread | midpoint | 1d change | vol24h | liquidity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| information_flow_watch | HSBC Championships, Qualification: Storm Hunter vs Aliaksandra Sasnovich | 5 | 22.613833 | 22.613833 | 0.0100 | 0.9950 | 0.6750 | 238892.48 | 62744.53 |
| information_flow_watch | LoL: KT Rolster vs Dplus KIA (BO5) - LCK Road to MSI | 5 | 20.584110 | 20.584110 | 0.0010 | 0.9995 | 0.4845 | 3735928.58 | 228390.79 |
| information_flow_watch | Will Kimi Antonelli win the 2026 F1 Monaco Grand Prix? | 5 | 20.272529 | 20.272529 | 0.0400 | 0.8500 | 0.5700 | 241579.38 | 20713.71 |
| information_flow_watch | Bitcoin Up or Down on June 7? | 5 | 19.119148 | 19.119148 | 0.0010 | 0.9895 | 0.4945 | 285041.08 | 66470.11 |
| information_flow_watch | Counter-Strike: Virtus.pro vs GenOne (BO3) - European Pro League Series 7 Group D | 5 | 15.426623 | 15.426623 | 0.0100 | 0.9950 | 0.3150 | 216577.21 | 85705.90 |
| information_flow_watch | Will the price of Bitcoin be above $60,000 on June 7? | 5 | 14.127678 | 14.127678 | 0.0010 | 0.9965 | 0.2615 | 208737.36 | 27637.08 |
| market_making_watch | Set Handicap: Zverev (-2.5) vs Cobolli (+2.5) | 5 | 11.864733 | 11.864733 | 0.1800 | 0.1500 | -0.2000 | 507074.90 | 12161.45 |

Interpretation:

- The top event-market rows persisted in all five samples.
- This still does not estimate the true probability of any event.
- The next validation is CLOB depth, adverse selection, and external event
  signal quality.

## Current Polymarket CLOB Depth

This checks visible CLOB depth for the top current microstructure monitor
markets.

| question | outcome | bid | ask | spread | bid depth 5c | ask depth 5c | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US x Iran permanent peace deal by June 15, 2026? | Yes | 0.0500 | 0.0600 | 0.0100 | 2872223.69 | 634130.59 | 634.0306 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 15, 2026? | No | 0.9400 | 0.9500 | 0.0100 | 634130.59 | 2872223.69 | 634.0306 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | Yes | 0.1400 | 0.1500 | 0.0100 | 311630.29 | 171204.83 | 171.1048 | visible depth exists near both sides |
| US x Iran permanent peace deal by June 30, 2026? | No | 0.8500 | 0.8600 | 0.0100 | 171204.83 | 325036.45 | 171.1048 | visible depth exists near both sides |
| Bitcoin Up or Down on June 7? | Yes | 0.9930 | 0.9950 | 0.0020 | 45703.00 | 19798.55 | 19.7786 | visible depth exists near both sides |
| Bitcoin Up or Down on June 7? | No | 0.0050 | 0.0070 | 0.0020 | 19798.55 | 45703.00 | 19.7786 | visible depth exists near both sides |

Interpretation:

- Some top information-flow rows are near-certain event markets with a thin
  opposite side. They are less useful for immediate execution research.
- The current depth-positive prediction-market candidate is `US x Iran
  permanent peace deal`, where public visible depth exists near both sides.
- This still does not prove edge. The next step is an event probability model
  or news-flow model for the same market.

## Current Prediction Market Paper Tickets

- MicroStrategy BTC purchase June 2-8: paper event-model candidate. The market
  has strong visible depth, tight spread, and high 24h volume. This needs an
  external information model based on filings, company announcements, and
  Bitcoin purchase reporting before any trade.
- Israel closes its airspace by June 30: paper geopolitical event-model
  candidate. Depth exists, but the market needs a news-flow model and latency
  checks.
- Iran airspace markets are active but currently too thin near top of book for
  paper priority.
- Sports rows with depth are treated as market-making research only until a
  dedicated sports model exists.
- Prediction-market tickets are not probability estimates. They only identify
  markets where an external model might be worth building.
