# Market Observations

This experiment collects market-native observations for alpha-os trading tasks.

It is not a feature store, not a catalog, and not alpha-os package API. The
purpose is to capture data that would plausibly be observable by a trading
system before turning it into model inputs.

Generated observation files are intentionally not committed.

## Scripts

- `collect_binance_market_observations.py`
  - fetches public Binance market observations from the spot symbol inventory
  - uses `data-api.binance.vision` for spot observations
  - attempts USD-M futures observations unless `--skip-futures` is set
  - uses `--sample-symbols` only for smoke runs
  - writes raw-ish JSONL files under an output directory
  - keeps feature engineering and strategy logic out of the collector

## Example

```bash
uv run python experiments/market_observations/collect_binance_market_observations.py \
  --quote-asset USDT \
  --days 7 \
  --interval 1h \
  --output-dir /tmp/alpha_os_market_observations/binance \
  --summary /tmp/alpha_os_market_observations/binance.md
```

For smoke runs, add `--sample-symbols 5 --sample-seed 0 --skip-futures`.
