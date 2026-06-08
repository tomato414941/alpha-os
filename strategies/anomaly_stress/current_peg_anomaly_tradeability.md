# Current Peg Anomaly Tradeability

This separates a peg anomaly from a currently routeable trade candidate. A missing pool match is not proof that a route does not exist; it means this current public snapshot did not validate execution yet.

| symbol | status | side | score | peg deviation | pool matches | best pool | pool reserve USD | pool vol 24h | yield conflicts | reason |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| USDY | peg_anomaly_mechanics_watch | paper_mechanics_check | 73.6230 | 0.132460 | 0 |  | 0 | 0 | 7 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| USYC | peg_anomaly_mechanics_watch | paper_mechanics_check | 69.3748 | 0.127497 | 0 |  | 0 | 0 | 2 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| USDai | peg_anomaly_mechanics_watch | paper_mechanics_check | 67.4298 | 0.008595 | 0 |  | 0 | 0 | 4 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| reUSD | peg_anomaly_mechanics_watch | paper_mechanics_check | 63.0752 | 0.081505 | 0 |  | 0 | 0 | 1 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| apxUSD | peg_anomaly_mechanics_watch | paper_mechanics_check | 61.8002 | -0.056003 | 0 |  | 0 | 0 | 1 | peg anomaly also appears in yield-risk rows, so the mechanism matters before any trade |
| pmUSD | peg_anomaly_stale_or_unrouted | no_trade_until_route | 36.3262 | -0.316312 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| satUSD | peg_anomaly_stale_or_unrouted | no_trade_until_route | 30.1708 | -0.008542 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
| FRAX | peg_anomaly_stale_or_unrouted | no_trade_until_route | 30.1128 | -0.005640 | 0 |  | 0 | 0 | 0 | material peg anomaly lacks a current route/depth confirmation in the joined public snapshots |
