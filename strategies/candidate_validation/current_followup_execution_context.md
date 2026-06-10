# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.895149 | 307667160 | 3.3597 | 69224 | 0.014446 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | broad_alpha_paper | 191.9081 | 0.109500 | 91716820 | 1.0288 | 15035 | 0.066510 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 2482797 | 6.7969 | 4301 | 0.232495 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | broad_alpha_paper | 82.8195 | 0.109500 | 39953322 | 0.1352 | 122835 | 0.008141 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1797889 | 1.4458 | 10360 | 0.096529 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | broad_alpha_paper | 74.4835 | 0.109500 | 993786125 | 0.1607 | 102178 | 0.009787 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | broad_alpha_paper | 41.0557 | -0.398650 | 2624994 | 3.0698 | 11553 | 0.086556 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | broad_alpha_paper | 38.2737 | 0.107772 | 2801245502 | 0.1594 | 4597660 | 0.000218 | tradable_context_ok | public venue context does not obviously block a small repeat |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
