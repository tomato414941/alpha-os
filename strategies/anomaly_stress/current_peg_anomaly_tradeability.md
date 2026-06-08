# Current Peg Anomaly Tradeability

This separates a peg anomaly from a currently routeable trade candidate. A missing pool match is not proof that a route does not exist; it means this current public snapshot did not validate execution yet.

| symbol | status | side | score | peg deviation | pool matches | best pool | pool reserve USD | pool vol 24h | yield conflicts | reason |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| USDY | peg_anomaly_mechanics_watch | paper_mechanics_check | 73.3743 | 0.127486 | 0 |  | 0 | 0 | 7 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| reUSD | peg_anomaly_mechanics_watch | paper_mechanics_check | 63.0599 | 0.081198 | 0 |  | 0 | 0 | 1 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| apxUSD | peg_anomaly_mechanics_watch | paper_mechanics_check | 61.9318 | -0.058635 | 0 |  | 0 | 0 | 1 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| USYC | peg_anomaly_mechanics_watch | paper_mechanics_check | 61.3602 | 0.127203 | 0 |  | 0 | 0 | 0 | large-supply peg anomaly needs issuer, redemption, and venue mechanics before execution |
| pmUSD | peg_anomaly_stale_or_unrouted | no_trade_until_route | 36.7598 | -0.337988 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| satUSD | peg_anomaly_stale_or_unrouted | no_trade_until_route | 30.1257 | -0.006287 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| FRAX | peg_anomaly_stale_or_unrouted | no_trade_until_route | 30.1205 | -0.006025 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| DOLA | peg_anomaly_stale_or_unrouted | no_trade_until_route | 30.1148 | -0.005738 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
