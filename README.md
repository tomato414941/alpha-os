# alpha-os

`alpha_os` is the current mainline package for signal discovery research,
strategy definition, and out-of-sample evaluation.

## Current Mainline

- package: `src/alpha_os/`
- entrypoint: `python -m alpha_os --help`
- focus:
  - signal discovery research
  - screening and compressed-belief artifacts
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

uv run python -m alpha_os show-report \
  --db "$DB"

uv run python -m alpha_os show-diagnostics \
  --db "$DB"
```

The checked-in test `tests/test_alpha_os_minimal_oos_workflow.py` verifies that
this path preserves fixed train/evaluation ranges, report contract fields,
decision traces, and candidate-vs-baseline promotion inputs.

## Minimal Fixed-State OOS Golden Path

This workflow also uses only checked-in fixture CSV data. It first materializes
a source initial strategy state, then creates a fixed-state replay evaluation
task and runs a strict OOS report.

```bash
DB=/tmp/alpha-os-minimal-fixed-state-oos.db
rm -f "$DB"

uv run python -m alpha_os apply-manifest \
  --manifest examples/minimal_fixed_state_oos.json \
  --db "$DB"

uv run python -m alpha_os run-walk-forward \
  --evaluation-spec-id minimal_fixed_state_train_eval \
  --db "$DB"

# Select the generated initial_strategy_state_id, then create the replay task:
uv run python -m alpha_os create-fixed-state-evaluation-task \
  --source-evaluation-task-id minimal_fixed_state_training_case \
  --source-initial-strategy-state-id <initial_strategy_state_id> \
  --evaluation-spec-id minimal_fixed_state_oos_eval \
  --db "$DB"

uv run python -m alpha_os run-walk-forward \
  --evaluation-spec-id minimal_fixed_state_oos_eval \
  --db "$DB"

uv run python -m alpha_os show-report \
  --db "$DB"
```

The checked-in test `tests/test_alpha_os_minimal_fixed_state_oos_workflow.py`
verifies strict OOS contract output and fixed-state provenance artifacts.

## Further Reading

- [DESIGN.md](DESIGN.md)
- [docs/design/README.md](docs/design/README.md)
