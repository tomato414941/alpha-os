# alpha-os

`alpha_os` is the current mainline package for trading strategy definition,
backtesting, and out-of-sample evaluation.

## Current Mainline

- package: `src/alpha_os/`
- focus:
  - trading strategy contracts
  - strategy backtests
  - OOS evaluation
  - portfolio decision flows

## Repository Layout

- `src/alpha_os/`
  - mainline strategy / evaluation runtime

## Development

```bash
uv sync --extra dev
```

## Testing

```bash
uv run ruff check src tests
uv run pytest -q
```

## Further Reading

- [docs/design/README.md](docs/design/README.md)
