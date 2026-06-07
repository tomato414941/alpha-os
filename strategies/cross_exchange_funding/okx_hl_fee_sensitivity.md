# OKX-Hyperliquid Fee Sensitivity

This is not a fee schedule. It is a paper sensitivity check.

Assumption: one entry and one exit on each venue.

| scenario | round-trip fee rate | 8h after fee | 8h USDT | 24h after fee | 24h USDT | survives 8h | survives 24h |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| very_low_fee | 0.00008 | 0.00001284 | 0.01284 | 0.00025603 | 0.25603 | True | True |
| low_fee | 0.0002 | -0.00010716 | -0.10716 | 0.00013603 | 0.13603 | False | True |
| one_bps_each | 0.0004 | -0.00030716 | -0.30716 | -0.00006397 | -0.06397 | False | False |
| two_bps_each | 0.0008 | -0.00070716 | -0.70716 | -0.00046397 | -0.46397 | False | False |
| five_bps_each | 0.002 | -0.00190716 | -1.90716 | -0.00166397 | -1.66397 | False | False |

## Interpretation

The BTC paper ticket is fee-sensitive. If both venues require more than roughly sub-bps effective execution per fill, the 8h edge disappears. This makes maker execution, rebates, or longer holding windows central to whether the candidate is real.
