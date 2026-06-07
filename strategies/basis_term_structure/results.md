# Basis Term Structure Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.basis_term_structure.current_deribit_futures_basis
```

Interpretation:

- positive annualized basis means the dated future is rich versus index
- negative annualized basis means the dated future is cheap versus index
- this is closer to carry / relative-value than directional price prediction
- any candidate still needs hedge route, fees, margin, funding, and depth checks

## Current Candidates

- `BTC-26MAR27`: rich dated future, short-basis watch.
- `BTC-25DEC26`: rich dated future, short-basis watch.
- `ETH-12JUN26`: cheap dated future, long-basis watch.
- `BTC-25SEP26`: rich dated future, short-basis watch.
- `ETH-26MAR27`: rich dated future, short-basis watch.

Near-expiry BTC/ETH rows can show larger annualized basis, but the current
screen marks weak volume or wide-spread rows as liquidity watches rather than
paper candidates.
