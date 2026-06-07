# Current BTC ETF Funding Paper Ticket

This compares current venues for the active BTC ETF-flow/funding paper watch. It is not a live trade instruction.

| venue | instrument | side | ann funding | carry side | volume USD | OI USD | spread/impact | basis/premium | score | status | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| OKX | BTC-USDT-SWAP | short | 0.00598260 | short_perp_receives_funding | 6477616971 | 1870447820 | 0.00000161 | -0.00058067 | 5.005241 | paper_venue_candidate | short carry, liquidity, and visible friction are acceptable for paper watch |
| Hyperliquid | BTC-USD perpetual | short | 0.10950000 | short_perp_receives_funding | 2175401667 | 2067460384 | 0.00018181 | -0.00038600 | 2.266335 | paper_venue_candidate | short carry, liquidity, and visible friction are acceptable for paper watch |

## Caveat

A paper ticket still needs account fee tier, maker/taker behavior, margin mode, stop execution, mark/index basis, and funding timestamp checks. The current score only ranks visible public venue context.
