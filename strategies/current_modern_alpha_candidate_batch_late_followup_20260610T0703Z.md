# Current Modern Alpha Candidate Batch Late Follow-Up

This is a delayed follow-up for the modern alpha candidate batch opened around
`2026-06-09T23:32Z`.

It is not an exact `1h` or `4h` score. The fixed horizons were missed, so this
file records a late checkpoint instead of pretending to be the primary result.

Checked at `2026-06-10T07:03Z`.

## Main Read

Validation should not block other work. This follow-up shows why: several
short-lived modern inputs changed regime before the delayed checkpoint. The
right operating model is to run checkpoints when due while continuing broader
discovery in parallel.

## Family Summary

| family | late read |
| --- | --- |
| liquidation + OI | Original top shocks mostly faded or changed action; do not promote the first batch's liquidation rows. |
| microstructure / order-flow | ADA short was favorable late; NEAR, OP, and HYPE longs were adverse; BTC and SUI were neutral. |
| attention/event | H long was adverse; WLD/ZEC/HYPE/BEAT were watch-only and should not be scored as trades. |
| prediction-market odds | BTC event hedges were neutral late; no promotion without event-to-asset attribution. |
| options / vol surface | BTC/ETH near-term put skew intensified; keep as volatility context, not naked direction. |
| wallet/entity flow | Not scored because source was stale. |
| cross-venue / basis | Not scored because source was stale. |
| legacy shallow screens | MORPHO and HYPE looked favorable late, but remain non-promotable because they lack required modern input. |

## Decisions

- Reject the current NEAR, OP, HYPE microstructure-long rows.
- Keep ADA microstructure-short as a lead only, because exact `1h/4h` was missed.
- Reject ZEC as a legacy funding/relative-strength candidate.
- Do not promote any stale wallet, cross-venue, or basis row.
- Keep options skew as a risk/hedge context lane, not as a standalone spot alpha.

## Process Change

Do not wait idle for validation windows. Future work should split into:

- checkpoint lane: record due `15m/1h/4h` outcomes when timestamps arrive;
- discovery lane: continue finding new modern-input candidates in parallel;
- refresh lane: freshen stale wallet/cross-venue/basis inputs before they can be
  scored.

Machine-readable rows are in
`strategies/current_modern_alpha_candidate_batch_late_followup_20260610T0703Z.csv`.
