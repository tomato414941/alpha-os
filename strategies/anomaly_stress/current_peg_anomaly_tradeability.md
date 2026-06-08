# Current Peg Anomaly Tradeability

This separates a peg anomaly from a currently routeable trade candidate. A missing pool match is not proof that a route does not exist; it means this current public snapshot did not validate execution yet.

| symbol | status | side | score | peg deviation | pool matches | best pool | pool reserve USD | pool vol 24h | yield conflicts | reason |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| USDY | peg_anomaly_mechanics_watch | paper_mechanics_check | 73.4305 | 0.128610 | 0 |  | 0 | 0 | 7 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| reUSD | peg_anomaly_mechanics_watch | paper_mechanics_check | 63.0674 | 0.081349 | 0 |  | 0 | 0 | 1 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| USYC | peg_anomaly_mechanics_watch | paper_mechanics_check | 61.3602 | 0.127203 | 0 |  | 0 | 0 | 0 | large-supply peg anomaly needs issuer, redemption, and venue mechanics before execution |
| apxUSD | peg_anomaly_mechanics_watch | paper_mechanics_check | 61.2946 | -0.045892 | 0 |  | 0 | 0 | 1 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| pmUSD | peg_anomaly_stale_or_unrouted | no_trade_until_route | 35.7374 | -0.286870 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| FRAX | peg_anomaly_stale_or_unrouted | no_trade_until_route | 30.1244 | -0.006221 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| DOLA | peg_anomaly_stale_or_unrouted | no_trade_until_route | 30.1239 | -0.006197 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| USDf | peg_anomaly_deprioritize | none | 20.0000 | -0.002271 | 0 |  | 0 | 0 | 0 | peg deviation is not material after route screening |
