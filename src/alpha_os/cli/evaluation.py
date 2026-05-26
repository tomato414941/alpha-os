from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "debug-finalize-observation": _legacy.cmd_finalize_observation,
    "debug-generate-evaluation-input": _legacy.cmd_generate_evaluation_input,
    "debug-generate-evaluation-inputs": _legacy.cmd_generate_evaluation_inputs,
    "debug-apply-backfill": _legacy.cmd_apply_backfill,
    "debug-apply-signal-candidates-backfill": _legacy.cmd_apply_signals_backfill,
    "run-evaluation": _legacy.cmd_run_evaluation,
    "run-walk-forward-evaluation": _legacy.cmd_run_walk_forward_evaluation,
    "run-walk-forward": _legacy.cmd_run_walk_forward_evaluation,
}
