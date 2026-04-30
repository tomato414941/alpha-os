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
- `examples/minimal_oos.json`
  - fixture-backed golden path for strict OOS evaluation

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
PYTHONPATH=src python -m alpha_os --help
```

## Minimal OOS Golden Path

This workflow uses only checked-in fixture CSV data. It does not require an
external data service.

```bash
DB=/tmp/alpha-os-minimal-oos.db
rm -f "$DB"

PYTHONPATH=src python -m alpha_os apply-runtime-manifest \
  --manifest examples/minimal_oos.json \
  --db "$DB"

PYTHONPATH=src python -m alpha_os run-walk-forward-evaluation \
  --evaluation-spec-id minimal_oos_eval \
  --db "$DB"

PYTHONPATH=src python -m alpha_os show-evaluation-report \
  --db "$DB"

PYTHONPATH=src python -m alpha_os show-evaluation-diagnostics \
  --db "$DB"
```

The checked-in test `tests/test_alpha_os_minimal_oos_workflow.py` verifies that
this path preserves fixed train/evaluation ranges, report contract fields,
decision traces, and candidate-vs-baseline promotion inputs.

## Further Reading

- [DESIGN.md](DESIGN.md)
- [docs/design/README.md](docs/design/README.md)
