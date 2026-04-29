# alpha-os

`alpha_os` is the current mainline package for signal discovery, strategy
definition, and strict out-of-sample evaluation.

## Current Mainline

- package: `src/alpha_os/`
- entrypoint: `python -m alpha_os --help`
- focus:
  - signal discovery
  - strategy specs and initial strategy state
  - strict OOS evaluation
  - portfolio decision and backtest flows

## Mainline CLI

```bash
python -m alpha_os --help

python -m alpha_os list-runtime-manifests

python -m alpha_os run-diagnostic-evaluation \
  --manifest fixture_daily_diagnostic \
  --evaluation-spec-id fixture_daily_diagnostic_eval \
  --details

python -m alpha_os run-walk-forward-evaluation \
  --evaluation-spec-id global_macro_futures_daily_trend_carry_eval

python -m alpha_os show-evaluation-report
```

Additional development and diagnostic commands exist for test and research
workflows, but they are intentionally hidden from the public CLI help.

The `fixture_daily_diagnostic` manifest uses only checked-in CSV fixtures under
`tests/fixtures/diagnostic_prices/`; it does not require `signal-noise` or any
external market-data API.

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
