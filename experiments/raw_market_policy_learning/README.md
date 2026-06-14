# Raw Market Policy Learning

This experiment probes whether raw market windows can support offline-to-online
policy learning.

It is not a trading strategy, not alpha-os package API, and not a claim of
tradable alpha. The goal is to keep human-designed alpha rules out of the first
pass: collect raw-ish market windows, attach future action rewards, then test
whether a sequence model can learn anything useful from that table.

Generated datasets and model outputs are intentionally not committed.

## Scripts

- `build_raw_window_tensor.py`
  - fetches public Binance hourly market data
  - builds `x = samples x instruments x lookback x raw_features`
  - builds `reward = samples x instruments x horizons x actions`
- `run_transformer_probe.py`
  - trains a small fixed Transformer encoder over the raw windows
  - compares the learned argmax policy with always-long, always-short, and
    always-flat baselines
- `fetch_signal_noise_streams.py`
  - lists signal-noise streams
  - fetches selected streams through the signal-noise batch API
  - keeps this private data adapter scoped to the experiment

## Example

```bash
uv run python experiments/raw_market_policy_learning/build_raw_window_tensor.py \
  --symbols 30 \
  --days 90 \
  --lookback 72 \
  --output /tmp/alpha_os_raw_window_tensor.npz \
  --summary /tmp/alpha_os_raw_window_tensor.md

uv run --with torch python experiments/raw_market_policy_learning/run_transformer_probe.py \
  --input /tmp/alpha_os_raw_window_tensor.npz \
  --summary /tmp/alpha_os_transformer_probe.md

uv run python experiments/raw_market_policy_learning/fetch_signal_noise_streams.py \
  --catalog-output /tmp/alpha_os_signal_noise_catalog.csv \
  --summary /tmp/alpha_os_signal_noise_probe.md \
  --name btc_ohlcv \
  --name funding_rate_btc \
  --since 2026-01-01 \
  --data-output /tmp/alpha_os_signal_noise_streams.csv
```

## Current Read

The first `/tmp` probe produced a valid tensor dataset and a working Transformer
training run, but the fixed model did not beat simple baselines. The next useful
step is to add more raw observable streams, such as funding, open interest,
liquidations, and order book state, rather than tuning this model.
