# Current Peg Anomaly Tradeability

This separates a peg anomaly from a currently routeable trade candidate. A missing pool match is not proof that a route does not exist; it means this current public snapshot did not validate execution yet.

| symbol | status | side | score | peg deviation | pool matches | best pool | pool reserve USD | pool vol 24h | yield conflicts | reason |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| USDY | peg_anomaly_mechanics_watch | paper_mechanics_check | 73.7786 | 0.135572 | 0 |  | 0 | 0 | 7 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| USYC | peg_anomaly_mechanics_watch | paper_mechanics_check | 69.3748 | 0.127497 | 0 |  | 0 | 0 | 2 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| reUSD | peg_anomaly_mechanics_watch | paper_mechanics_check | 63.0823 | 0.081645 | 0 |  | 0 | 0 | 1 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| apxUSD | peg_anomaly_mechanics_watch | paper_mechanics_check | 60.8208 | -0.036416 | 0 |  | 0 | 0 | 1 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| pmUSD | peg_anomaly_stale_or_unrouted | no_trade_until_route | 36.5369 | -0.326846 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| DOLA | peg_anomaly_stale_or_unrouted | no_trade_until_route | 30.1149 | -0.005747 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| USDf | peg_anomaly_deprioritize | none | 20.0000 | -0.003058 | 0 |  | 0 | 0 | 0 | peg deviation is not material after route screening |
| MSUSD | peg_anomaly_deprioritize | none | 20.0000 | -0.003157 | 0 |  | 0 | 0 | 1 | peg deviation is not material after route screening |
