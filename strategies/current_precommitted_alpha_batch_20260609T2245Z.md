# Current Precommitted Alpha Batch

This is a one-off candidate batch, not a queue, tracker, registry, or framework.
It exists to avoid single-name overfitting and to force a broad search before
looking at outcomes.

Observed:

- Crypto perp anchors: `2026-06-09T22:45:18Z`.
- Equity, ETF, commodity, and rates-proxy anchors: `2026-06-09T22:35:39Z`
  from the prior cross-market event follow-up. Fresh Yahoo chart fetch was
  unstable, so rows without a fresh anchor are marked as `paper_watch`.

## Fixed Scoring Rule

- Primary horizon: `1h`.
- Secondary horizon: `4h`.
- Path check: `15m` adverse excursion only. A clean 15m print does not promote a
  candidate by itself.
- Crypto perps: score return relative to a BTC/ETH basket, minus spread and a
  simple taker-cost proxy. Funding direction is part of the entry context, not a
  reason to rewrite the thesis after the fact.
- Equity and ETF rows: score relative to the stated reference, usually QQQ,
  SPY, BTC, USO, or rates/commodity confirmation.
- Macro rows are conditional: the condition must be known before scoring.
- A single winner is not an alpha. Treat it as a lead until at least five
  comparable candidates in the same rule family have been observed.

## Batch Shape

| group | count | purpose |
| --- | ---: | --- |
| Crypto funding / carry / relative | 18 | Avoid making ZEC the only story; test both long funding dislocations and crowded shorts. |
| Equity / crypto-equity / semis | 6 | Check whether the same risk event is better expressed outside tokens. |
| Commodity / rates / macro | 7 | Keep oil, gold, rates, and CPI/Hormuz paths in the same batch. |

Total candidates: 31.

## Highest-Signal Rows

| id | action | anchor | why it is in the batch |
| --- | --- | ---: | --- |
| zec-carry-relative-long-20260609T2245Z | long ZEC | 439.2900 | Prior survivor, still negative funding, liquid enough for a small paper repeat. |
| morpho-carry-strength-long-20260609T2245Z | long MORPHO | 1.9001 | Large negative funding and positive 24h move; checks whether ZEC is not unique. |
| hype-relative-weakness-short-20260609T2245Z | short HYPE | 57.8170 | Tests residual weakness with strong liquidity, but path risk remains important. |
| zro-relative-weakness-short-20260609T2245Z | short ZRO | 0.8465 | Positive funding, weaker setup, and known prior unlock-thesis failure forces a cleaner relative test. |
| sol-clean-beta-long-20260609T2245Z | long SOL | 65.1350 | Liquid beta candidate with negative funding; useful as a cleaner crypto benchmark. |
| semis-soft-cpi-continuation-long-20260609T2235Z | long SMH | 591.0100 | Tests whether risk-on is better expressed through semis than crypto. |
| mstr-residual-weakness-short-20260609T2235Z | short MSTR | 117.0200 | Crypto-equity residual weakness was visible in MSTR, not broadly in COIN/HOOD. |
| uso-hormuz-oil-long-20260609T2235Z | long USO | 131.3000 | Oil rebounded; still needs independent Hormuz/shipping evidence before stronger action. |
| cpi-soft-risk-on-basket-20260609T2235Z | conditional long risk basket | mixed | Keeps CPI as an event condition instead of turning it into a narrative after the move. |
| cpi-hot-risk-off-basket-20260609T2235Z | conditional short risk basket | mixed | Same event, opposite condition; prevents only keeping the direction that later worked. |

## Guardrails

- Do not select winners by changing horizons after the fact.
- Do not turn a liquidity warning into a tradable idea without a fresh anchor.
- Do not promote a macro narrative unless the concrete instrument and
  invalidation were present before the outcome.
- Do not create a new research framework from this batch. The next action is to
  score these rows at the fixed horizons, then discard weak families quickly.

The machine-readable rows are in
`strategies/current_precommitted_alpha_batch_20260609T2245Z.csv`.
