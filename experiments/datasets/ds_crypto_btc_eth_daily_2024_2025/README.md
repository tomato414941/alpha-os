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

The source provided daily `close` rows with `volume`, and sparse
`open_interest`.

The source did not provide `funding_rate` for `BTCUSDT` or `ETHUSDT` through the
public observation contract checked here. The `funding_rate` column is present
but empty.

This dataset is not sufficient to fully judge the hypothesis until funding rate
data is supplied and the open interest frequency decision is made, or the
hypothesis is narrowed to exclude those inputs.

## Validation

- `BTCUSDT.csv`: 731 rows, 0 missing `close`, 0 missing `volume`, 731 missing
  `funding_rate`, 626 missing `open_interest`
- `ETHUSDT.csv`: 731 rows, 0 missing `close`, 0 missing `volume`, 731 missing
  `funding_rate`, 626 missing `open_interest`
