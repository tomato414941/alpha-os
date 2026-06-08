# Current BTC ETF Funding Paper Ticket

This compares current venues for the active BTC ETF-flow/funding paper watch. It is not a live trade instruction.

| venue | instrument | side | ann funding | carry side | volume USD | OI USD | spread/impact | basis/premium | score | status | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| OKX | BTC-USDT-SWAP | short | -0.08956514 | long_perp_receives_funding | 9501643525 | 1863677996 | 0.00000781 | -0.00053716 | 4.909117 | reject | short side does not receive funding |
| Hyperliquid | BTC-USD perpetual | short | 0.10950000 | short_perp_receives_funding | 3327338377 | 1930148299 | 0.00013585 | -0.00042147 | 3.422832 | paper_venue_candidate | short carry, liquidity, and visible friction are acceptable for paper watch |

## Caveat

A paper ticket still needs account fee tier, maker/taker behavior, margin mode, stop execution, mark/index basis, and funding timestamp checks. The current score only ranks visible public venue context.
