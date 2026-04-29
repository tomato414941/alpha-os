# alpha-os

`alpha_os` is the current mainline package for signal discovery research,
strategy definition, and out-of-sample evaluation.

## Current Mainline

- package: `src/alpha_os/`
- entrypoint: `python -m alpha_os --help`
- focus:
  - signal discovery research
  - strategy specs and initial strategy state
  - OOS evaluation
  - portfolio decision and evaluation flows

## Repository Layout

- `src/alpha_os/`
  - mainline discovery / strategy / evaluation runtime
- `config/runtime_manifests/`
  - evaluation manifest examples

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install the optional data-service integration only when a compatible
`signal-noise` package is available:

```bash
pip install -e ".[data]"
```

## Testing

```bash
ruff check src tests
PYTHONPATH=src pytest -q
```

## Further Reading

- [DESIGN.md](DESIGN.md)
- [docs/design/README.md](docs/design/README.md)
