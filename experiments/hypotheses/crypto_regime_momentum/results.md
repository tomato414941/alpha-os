# Crypto Regime Momentum Results

Status: promising but sparse.

## Evidence

- Candidate beats the baseline overall:
  - baseline total net return: -20.2%
  - candidate total net return: +68.9%
- Candidate wins in both calendar periods:
  - 2024 candidate total net return: +51.0%
  - 2025 candidate total net return: +11.9%
- Candidate wins for both assets:
  - BTCUSDT candidate total net return: +27.3%
  - ETHUSDT candidate total net return: +54.9%
- Candidate wins under shifted evaluation start dates:
  - 2024-03-01 start: +68.9%
  - 2024-04-01 start: +68.9%
  - 2024-05-01 start: +66.9%
  - 2024-06-01 start: +45.7%
- Candidate remains ahead of the baseline across tested cost levels:
  - 0 bps: baseline -11.7%, candidate +79.5%
  - 5 bps: baseline -20.2%, candidate +68.9%
  - 10 bps: baseline -27.8%, candidate +59.0%
  - 25 bps: baseline -46.7%, candidate +32.6%
  - 50 bps: baseline -67.8%, candidate -2.1%
- The 30 day trend filter matters. Removing it reduces total net return to
  -10.2%.
- The funding filter matters. Removing it reduces total net return to +39.6%.

## Cautions

- Candidate is sparse:
  - invested days: 125 / 639
  - flat days: 514 / 639
- Candidate turns slightly negative at 50 bps cost:
  - candidate total net return: -2.1%
  - baseline total net return: -67.8%
- Open interest did not contribute to the first candidate.
- Volatility scaling reduced performance in the first candidate.

## Current Decision

Continue. The first candidate beats the baseline across the tested period,
assets, shifted start dates, and cost levels, but it is sparse and does not
remain profitable at 50 bps.

## Next

Check whether the same rule remains useful on a broader crypto universe or a
later unseen period before moving it into alpha-os runtime evaluation.
