# Manual Paper Log

Record daily manual paper decisions here.

This log is for observing the strategy's behavior. It is not an execution
system and does not imply that trades were placed.

## Template

```text
date=
strategy=crypto
variant=
mode=manual_paper
target_weights:
notes:
```

## 2025-12-31 Snapshot

```text
date=2025-12-31T00:00:00+00:00
strategy=crypto
variant=7d_momentum
mode=manual_paper
target_weights:
  ETHUSDT: 1.000000
notes:
  Generated from the checked-in historical dataset, not live market data.
  This used the initial 7 day momentum baseline.
```

## 2025-12-31 Current Baseline Snapshot

```text
date=2025-12-31T00:00:00+00:00
strategy=crypto
variant=7d_momentum_30d_trend
mode=manual_paper
target_weights:
  ETHUSDT: 1.000000
notes:
  Generated from the checked-in historical dataset, not live market data.
  Current baseline uses the 7 day momentum + 30 day trend filter.
```
