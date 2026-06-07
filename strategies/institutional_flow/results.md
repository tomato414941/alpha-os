# Institutional Flow Results

This lane was added to cover slower institutional demand, starting with Bitcoin
spot ETF flows.

Initial result:

- Bitbo exposes daily BTC ETF flow history in page data.
- The page data must be aggregated by date because the embedded rows include
  same-day issuer-level entries.
- BTC ETF flow was labeled from the day after the flow date to avoid using
  same-day price movement.
- Labeled observations: 555
- Mean directional 1d: 0.00233183
- Mean directional 3d: 0.00742992
- Mean directional 5d: 0.01268852
- 5d hit rate: 0.5730

Current read:

- ETF inflow as long BTC context and ETF outflow as short BTC context has
  positive coarse 5-day directional behavior in this panel.
- This is not deployable PnL. It excludes funding PnL, intraday timing,
  transaction costs, and regime splits.
- The next useful test is whether ETF flow improves after splitting by perp
  funding alignment/divergence and large-flow thresholds.

Regime split:

- `large_5d_outflow`: observations 53, mean directional 5d 0.03437137,
  hit rate 0.7170.
- `btc_etf_distribution_label`: observations 89, mean directional 5d
  0.01862256, hit rate 0.5843.
- `btc_etf_inflow_context_label`: observations 139, mean directional 5d
  0.01637483, hit rate 0.6187.
- `large_5d_inflow`: observations 226, mean directional 5d 0.01522096,
  hit rate 0.5796.

Current best candidate is not "ETF flow always works". It is that large
rolling ETF outflow, treated as short BTC regime context, has the strongest
5-day directional label in this first pass.

Funding split:

- Funding alignment is based on the label-start funding rate, not future
  funding.
- `large_5d_outflow__funding_aligned`: observations 43, mean directional 5d
  plus funding 0.03263416, mean start funding support 0.00014644, hit rate
  0.6977.
- `large_5d_outflow`: observations 51, mean directional 5d plus funding
  0.03102419, mean start funding support 0.00010456, hit rate 0.6863.
- The useful current hypothesis is narrower: large rolling ETF outflow as a
  BTC short context is more interesting when BTCUSDT perp funding is positive,
  because the short direction also receives funding.
- This still is not deployable PnL. It excludes intraday entry timing,
  drawdown behavior, liquidity, fee tier, and execution assumptions.

Paper rule pass:

- Rule: enter BTC short after large rolling ETF outflow when label-start
  BTCUSDT funding support is positive; hold 5 days; skip overlapping signals.
- Fee assumption: 5 bps per side.
- Trades: 21, skipped overlap signals: 22.
- Total return: 0.78885853.
- Mean net 5d return: 0.02961324.
- Hit rate: 0.6190.
- Max drawdown: -0.07880819.
- This is the first institutional-flow candidate that survived non-overlap and
  a simple fee haircut. The next risk is whether intraday timing, mark/index
  basis, liquidation buffer, and actual account fees erase it.

Robustness pass:

- Fee sensitivity still survives from 1 bps/side through 50 bps/side.
- At 50 bps/side, total return is 0.48684751, hit rate is 0.5714, and max
  drawdown is -0.09600412.
- Calendar-year split is positive in 2024, 2025, and 2026, but the sample is
  still thin: 2024 has 3 trades, 2025 has 10 trades, and 2026 has 8 trades.
- Entry-delay sensitivity also remains positive: 1-day delay total return is
  0.36179234, and 2-day delay total return is 0.54650100.
- The main weakness is not fee sensitivity. It is sample size, drawdown under
  delayed entry, and execution realism.
