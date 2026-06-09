# Current Precommitted Alpha Batch Path Check

This is the precommitted `15m` path check for the broad alpha batch opened at
`2026-06-09T22:45:18Z`.

It is not the primary outcome. The primary horizon remains `1h`; the secondary
horizon remains `4h`.

Checked at `2026-06-09T23:05:55Z`.

## Reference Move

| reference | entry | current | move |
| --- | ---: | ---: | ---: |
| BTC | 61846.0000 | 61726.0000 | -19.40 bps |
| ETH | 1645.8000 | 1638.9000 | -41.93 bps |
| BTC/ETH basket | mixed | mixed | -30.66 bps |

## Early Read

| read | count | candidates |
| --- | ---: | --- |
| early adverse move | 6 | ZEC, INJ, OP, ZRO, XMR, VVV |
| early favorable but not promotion | 3 | NEAR, WLD, LIT |
| cost/depth warning but not promotion | 1 | FET |
| neutral path check | 8 | MORPHO, MOVE, SOL, APT, SEI, HYPE, SUI, RUNE |

The count intentionally does not select winners. `15m` was predefined only as
an adverse-excursion check.

## Important Reads

- ZEC reversed sharply after the batch anchor: `-310.34 bps` relative to the
  BTC/ETH basket. This blocks any immediate promotion even though it was a prior
  survivor.
- INJ, OP, ZRO, XMR, and VVV also showed early adverse paths.
- NEAR, WLD, and LIT were favorable early, but the rule explicitly forbids
  promotion from `15m`.
- FET was directionally favorable but current spread was too high to promote.
- ETF, equity, commodity, and rates-proxy rows are not scored here because the
  fresh realtime fetch was unstable and the primary `1h` horizon has not yet
  elapsed.

## Next Action

At or after `2026-06-09T23:45:18Z`, run the `1h` primary scoring for the full
batch. Do not change horizons, metrics, or candidate descriptions based on this
early check.

Machine-readable rows are in
`strategies/current_precommitted_alpha_batch_path_check_20260609T2305Z.csv`.
