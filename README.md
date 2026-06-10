# alpha-os

`alpha_os` is a small package for defining trading strategy contracts.

## Current Mainline

- package: `src/alpha_os/`
- focus:
  - `TradingStrategy` as the black-box decision contract
  - concrete examples under `examples/`
  - minimal tests for the maintained contract and examples

## Repository Layout

- `src/alpha_os/`
  - maintained package code
- `examples/`
  - concrete trading strategy sketches; not package API
- `experiments/`
  - frozen historical research snapshot; do not add new experiments

## Development

```bash
uv sync --extra dev
```

## Testing

```bash
uv run ruff check .
uv run pytest -q
```

## Further Reading

- [docs/design/README.md](docs/design/README.md)
