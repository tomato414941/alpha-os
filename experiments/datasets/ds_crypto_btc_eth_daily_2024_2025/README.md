# Crypto BTC/ETH Daily Dataset

Purpose: evidence data candidate for crypto BTC/ETH daily experiments,
including `hyp_5880aba5` / Crypto Regime Momentum.

Status: usable for first hypothesis check.

## Source

- Source: internal `signal-noise` observation API and Binance public data archive
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

The source provides daily `close`, `volume`, `funding_rate`, and
`open_interest` rows.

`funding_rate` uses the current `signal-noise` daily scalar observation output.
That is enough for feature experimentation, but not yet a final accounting
definition for funding cost or carry PnL.

`open_interest` uses the last Binance public archive metrics value for each UTC
day. That is enough for feature experimentation, but not a final execution or
accounting definition.

## Validation

- `BTCUSDT.csv`: 731 rows, 0 missing `close`, 0 missing `volume`, 0 missing
  `funding_rate`, 0 missing `open_interest`
- `ETHUSDT.csv`: 731 rows, 0 missing `close`, 0 missing `volume`, 0 missing
  `funding_rate`, 0 missing `open_interest`
