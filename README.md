# alpha-os

`alpha_os` is the current mainline package for signal discovery research,
strategy definition, and out-of-sample evaluation.

## Current Mainline

- package: `src/alpha_os/`
- focus:
  - signal discovery research
  - screening and compressed-belief artifacts
  - strategy specs and strategy checkpoints
  - OOS evaluation
  - portfolio decision and evaluation flows

## Repository Layout

- `src/alpha_os/`
  - mainline discovery / strategy / evaluation runtime
- `config/runtime_manifests/`
  - evaluation manifest examples
- `examples/minimal_oos.json`
  - fixture-backed golden path for strict OOS evaluation

## Development

```bash
uv sync --extra dev
```

Install the optional data-service integration only when a compatible
`signal-noise` package is available:

```bash
uv sync --extra dev --extra data
```

## Testing

```bash
uv run ruff check src tests
uv run pytest -q
```

## Further Reading

- [docs/design/README.md](docs/design/README.md)
