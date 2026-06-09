# Current Cross-Market Alpha Paper Entries

These are paper entries for falsification, not trade instructions.

Opened at `2026-06-09T16:14:50+00:00`.

| entry | candidate | decision | entry | hedge/reference | why opened | checkpoints |
| --- | --- | --- | ---: | --- | --- | --- |
| zec-carry-relative-long-20260609T161450Z | ZEC carry plus relative strength | paper long ZEC | 429.650000000000 | BTC 61028.000000000000 / ETH 1630.100000000000 | ZEC is less weak than BTC/ETH during the risk-off move, has negative funding, large OI/volume, tight spread, and usable depth. | 15m,1h,4h |
| hype-relative-weakness-short-20260609T161450Z | HYPE relative weakness after strength reversal | paper short HYPE | 59.011000000000 | BTC 61028.000000000000 / ETH 1630.100000000000 | The earlier HYPE long thesis is invalidated by the fresh snapshot: HYPE is now materially weaker than BTC/ETH, while short funding is not hostile. | 15m,1h,4h |

## First Checks

- Score absolute return and BTC/ETH-relative return.
- Include spread, taker fee, funding, depth, and adverse excursion.
- Reject ZEC if it underperforms BTC/ETH after costs or funding flips materially
  positive.
- Reject HYPE short if it recovers relative to BTC/ETH after costs or funding
  support disappears.
