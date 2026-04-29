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
  - executable evaluation setups, not the sole runtime truth
  - include observables, signal specs, subject sets, strategy specs, evaluation protocols, and evaluation cases
  - `global_macro_futures_daily_trend_carry.json` is the current cross-asset reference shape
  - ETF manifests remain as narrow examples, not the architectural center

## Legacy Boundary

The old `alpha_os_recovery` runtime has been removed from this repo.

This repo no longer carries:

- legacy paper/runtime CLI entrypoints
- legacy replay/admission runtime flows
- legacy deploy units
- legacy legacy-test suite
- legacy DSL-backed signal runtime

New runtime design belongs in `src/alpha_os/`.

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

- [AGENTS.md](AGENTS.md)
- [DESIGN.md](DESIGN.md)
- [docs/design/README.md](docs/design/README.md)
