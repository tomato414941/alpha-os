# Current Cross-Market Event Entries

These are conditional paper entries for falsification, not trade instructions.
They keep the search wider than crypto perps without adding tracking
infrastructure.

Observed at `2026-06-09T16:25:00+00:00`.

## Anchors

| instrument | anchor | note |
| --- | ---: | --- |
| QQQ | 693.1900 | weak intraday into CPI/FOMC |
| SMH | 563.6000 | weaker than QQQ; semiconductor pressure |
| NVDA | 201.4400 | AI beta |
| AVGO | 376.3800 | AI beta |
| TLT | 85.1150 | rates proxy |
| GLD | 391.1800 | gold proxy |
| USO | 128.8800 | oil proxy |
| COIN | 152.2100 | crypto equity beta |
| HOOD | 80.4000 | crypto/prediction-market equity beta |
| MSTR | 115.8800 | BTC treasury premium beta |
| BTC | 61028.0000 | crypto beta reference |
| ETH | 1630.1000 | crypto beta reference |

## Conditional Entries

| entry | condition | paper action | first check |
| --- | --- | --- | --- |
| cpi-hot-risk-off-20260609T1625Z | CPI is hot enough to reduce cut odds or raise no-cut/hike expectations | short QQQ/SMH/NVDA/AVGO/BTC/ETH; optional short TLT | 30m/2h post-release move vs SPY and BTC/ETH beta; TLT should confirm rates pressure |
| cpi-soft-risk-on-20260609T1625Z | CPI is soft enough to revive cut expectations | long SMH/QQQ/NVDA/AVGO; optional long BTC/ETH | 30m/2h bounce; require SMH breadth beyond one mega-cap |
| hormuz-oil-risk-20260609T1625Z | Hormuz disruption odds rise or shipping evidence worsens while oil rebounds | long USO; optional short QQQ/BTC beta | USO should lead or coincide with renewed risk pressure |
| semis-breakdown-continuation-20260609T1625Z | SMH remains weak into CPI and CPI is not dovish | short SMH/NVDA/AVGO vs QQQ/SPY reference | SMH relative weakness must persist after CPI |
| crypto-equity-beta-stress-20260609T1625Z | COIN/HOOD/MSTR underperform BTC/QQQ during crypto risk-off | short crypto equity residual weakness | residual weakness must remain after controlling for BTC and QQQ |

## Rejection Rules

- Reject macro entries if the event reaction contradicts the condition in the
  first 30 minutes.
- Reject equity-beta entries if they are only generic QQQ or BTC beta.
- Reject oil/Hormuz entries if oil remains weak or shipping-normalization
  evidence improves.

## Follow-Up

Checked at `2026-06-09T22:35:39+00:00`.

| instrument | anchor | follow-up | move bps | read |
| --- | ---: | ---: | ---: | --- |
| QQQ | 693.1900 | 707.8300 | 211.05 | risk beta rebounded |
| SMH | 563.6000 | 591.0100 | 486.34 | semis outperformed QQQ from the anchor |
| NVDA | 201.4400 | 208.1900 | 335.09 | AI beta rebounded |
| AVGO | 376.3800 | 392.1600 | 419.26 | AI beta rebounded |
| TLT | 85.1150 | 85.1200 | 0.59 | rates proxy flat from the anchor |
| GLD | 391.1800 | 390.7800 | -10.23 | gold did not confirm stress |
| USO | 128.8800 | 131.3000 | 187.77 | oil rebounded |
| COIN | 152.2100 | 155.5000 | 216.15 | roughly QQQ-like from the anchor |
| HOOD | 80.4000 | 83.7700 | 419.15 | outperformed QQQ from the anchor |
| MSTR | 115.8800 | 117.0200 | 98.38 | still weaker than QQQ from the anchor |

Pre-event read: the semiconductor breakdown continuation is not supported from
the anchor. Hormuz/oil remains worth watching because USO rebounded, but it
still needs independent shipping or prediction-market evidence. Crypto-equity
residual weakness is only visible in MSTR, not broadly in COIN/HOOD.
