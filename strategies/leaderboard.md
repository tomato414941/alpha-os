# Strategy Leaderboard

Command:

```text
uv run python -m strategies.leaderboard --format markdown
```

Result:

```text
| group | candidate | selection | total | sharpe | drawdown | turnover | best total bh | excess total | best sharpe bh | excess sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equity_index | top_momentum_126_252 | fixed_etf_universe | 0.562242 | 1.729854 | -0.192095 | 0.143258 | GLD 0.614538 | -0.052296 | GLD 1.775926 | -0.046072 |
| cash_rotation | risk_on_off_252 | fixed_etf_universe | 0.272188 | 1.135007 | -0.260208 | 0.014045 | GLD 0.614538 | -0.342350 | GLD 1.775926 | -0.640919 |
| cash_rotation | risk_on_off_126 | fixed_etf_universe | 0.367430 | 1.105907 | -0.135577 | 0.043568 | GLD 0.817698 | -0.450268 | GLD 1.773026 | -0.667119 |
| crypto | 7d_momentum_30d_trend_skfolio_max_ratio | manual_same_period_exclusion | 1.928955 | 1.069576 | -0.550211 | 0.327607 | XRPUSDT 1.181204 | 0.747751 | XRPUSDT 0.803036 | 0.266540 |
| equity_index | top_momentum_21_63 | fixed_etf_universe | 0.293484 | 0.801077 | -0.202137 | 0.284404 | GLD 0.862555 | -0.569071 | GLD 1.678484 | -0.877408 |
| equity_index | top_momentum_63_126 | fixed_etf_universe | 0.243337 | 0.727686 | -0.200477 | 0.155602 | GLD 0.817698 | -0.574361 | GLD 1.773026 | -1.045339 |
| crypto | 7d_momentum_30d_trend_skfolio_max_ratio | fixed_expanded_universe | 0.217615 | 0.444806 | -0.704457 | 0.389673 | XRPUSDT 1.181204 | -0.963589 | XRPUSDT 0.803036 | -0.358230 |
| cash_rotation | risk_on_off_63 | fixed_etf_universe | 0.023247 | 0.188761 | -0.304261 | 0.122936 | GLD 0.862555 | -0.839308 | GLD 1.678484 | -1.489723 |
| crypto | 7d_momentum_30d_trend_skfolio_max_ratio_eligible | rolling_asset_quality | -0.160001 | 0.145100 | -0.713836 | 0.297712 | XRPUSDT 1.322403 | -1.482405 | XRPUSDT 0.922692 | -0.777592 |
| crypto | 7d_momentum_30d_trend | fixed_expanded_universe | -0.322751 | -0.024984 | -0.712941 | 0.458219 | XRPUSDT 1.181204 | -1.503955 | XRPUSDT 0.803036 | -0.828020 |
| crypto | 7d_momentum_30d_trend_skfolio_min_variance | fixed_expanded_universe | -0.277097 | -0.043475 | -0.583201 | 0.485245 | XRPUSDT 1.181204 | -1.458301 | XRPUSDT 0.803036 | -0.846511 |
| crypto_pair_spread | zscore_pair_spread_1.5 | fixed_pairs | -0.426867 | -0.984258 | -0.455923 | 0.260291 | XRPUSDT 0.825711 | -1.252578 | XRPUSDT 0.723045 | -1.707303 |
| crypto_pair_spread | zscore_pair_spread_2 | fixed_pairs | -0.380166 | -1.038291 | -0.425759 | 0.164649 | XRPUSDT 0.825711 | -1.205877 | XRPUSDT 0.723045 | -1.761336 |
| crypto_pair_spread | zscore_pair_spread_1 | fixed_pairs | -0.620756 | -1.528876 | -0.636395 | 0.294189 | XRPUSDT 0.825711 | -1.446467 | XRPUSDT 0.723045 | -2.251921 |
```

Interpretation:

The only row that beats the best same-window buy-and-hold benchmark is the
manual crypto smaller-universe result, and that row is marked
`manual_same_period_exclusion` because it is selection-biased.

The best non-leaky candidate is `equity_index/top_momentum_126_252`. It nearly
matches `GLD` buy-and-hold on total return and Sharpe, but does not beat it.

Current broad conclusion:

- no non-leaky candidate clearly beats its best buy-and-hold benchmark yet
- ETF momentum is the closest non-leaky candidate
- cash rotation has attractive drawdown, but gives up too much return versus
  `GLD`
- crypto pair spread is not currently promising
- crypto manual smaller universe remains a diagnostic clue, not a credible
  result
