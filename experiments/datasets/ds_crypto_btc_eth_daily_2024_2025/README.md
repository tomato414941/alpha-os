# Crypto BTC/ETH Daily Dataset

Purpose: evidence data candidate for crypto BTC/ETH daily experiments,
including `hyp_5880aba5` / Crypto Regime Momentum.

Status: incomplete.

## Source

- Source: internal `signal-noise` observation API
- Retrieved at: 2026-04-30
- Assets: `BTCUSDT`, `ETHUSDT`
- Period: 2024-01-01 through 2025-12-31

## Files

- `BTCUSDT.csv`
- `ETHUSDT.csv`

## Columns

- `timestamp`
- `close`
- `volume`
- `funding_rate`
- `open_interest`

## Current Gap

The source provides daily `close`, `volume`, and `funding_rate` rows.

`funding_rate` uses the current `signal-noise` daily scalar observation output.
That is enough for feature experimentation, but not yet a final accounting
definition for funding cost or carry PnL.

`open_interest` remains sparse. This dataset is not sufficient to fully judge
the hypothesis until the open interest frequency decision is made, or the
hypothesis is narrowed to exclude that input.

## Validation

- `BTCUSDT.csv`: 731 rows, 0 missing `close`, 0 missing `volume`, 0 missing
  `funding_rate`, 626 missing `open_interest`
- `ETHUSDT.csv`: 731 rows, 0 missing `close`, 0 missing `volume`, 0 missing
  `funding_rate`, 626 missing `open_interest`
