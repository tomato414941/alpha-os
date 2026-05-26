# alpha-os

`alpha_os` is the current mainline package for signal discovery research,
strategy definition, and out-of-sample evaluation.

## Current Mainline

- package: `src/alpha_os/`
- entrypoint: `python -m alpha_os --help`
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
uv run python -m alpha_os --help
```

## Minimal OOS Golden Path

This workflow uses only checked-in fixture CSV data. It does not require an
external data service.

```bash
DB=/tmp/alpha-os-minimal-oos.db
rm -f "$DB"

uv run python -m alpha_os apply-manifest \
  --manifest examples/minimal_oos.json \
  --db "$DB"

uv run python -m alpha_os run-walk-forward \
  --evaluation-spec-id minimal_oos_eval \
  --db "$DB"
```

The checked-in test `tests/test_alpha_os_minimal_oos_workflow.py` verifies that
this path preserves fixed train/evaluation ranges and report contract fields.

## Minimal Fixed-State OOS Golden Path

This workflow also uses only checked-in fixture CSV data. It first materializes
a source strategy checkpoint, then runs a strict OOS report with that checkpoint
as an explicit evaluation input.

```bash
DB=/tmp/alpha-os-minimal-fixed-state-oos.db
rm -f "$DB"

uv run python -m alpha_os apply-manifest \
  --manifest examples/minimal_fixed_state_oos.json \
  --db "$DB"

uv run python -m alpha_os run-walk-forward \
  --evaluation-spec-id minimal_fixed_state_train_eval \
  --db "$DB"
```

The checked-in test `tests/test_alpha_os_minimal_fixed_state_oos_workflow.py`
verifies the checkpoint replay path, strict OOS contract output, and strategy
checkpoint provenance artifacts without requiring a manual checkpoint-linking CLI
step.

## Further Reading

- [docs/design/README.md](docs/design/README.md)
