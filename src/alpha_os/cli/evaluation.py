from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "debug-record-prediction": _legacy.cmd_record_prediction,
    "debug-finalize-observation": _legacy.cmd_finalize_observation,
    "debug-update-state": _legacy.cmd_update_state,
    "debug-generate-evaluation-input": _legacy.cmd_generate_evaluation_input,
    "debug-generate-evaluation-inputs": _legacy.cmd_generate_evaluation_inputs,
    "debug-apply-evaluation": _legacy.cmd_apply_evaluation,
    "debug-apply-evaluations": _legacy.cmd_apply_evaluations,
    "debug-apply-backfill": _legacy.cmd_apply_backfill,
    "debug-apply-signal-candidates-backfill": _legacy.cmd_apply_signals_backfill,
    "run-evaluation": _legacy.cmd_run_evaluation,
    "run-walk-forward-evaluation": _legacy.cmd_run_walk_forward_evaluation,
    "run-walk-forward": _legacy.cmd_run_walk_forward_evaluation,
    "create-fixed-state-evaluation-task": _legacy.cmd_create_fixed_state_evaluation_task,
    "show-evaluation-report": _legacy.cmd_show_evaluation_report,
    "show-report": _legacy.cmd_show_evaluation_report,
    "show-evaluation-diagnostics": _legacy.cmd_show_evaluation_diagnostics,
    "show-diagnostics": _legacy.cmd_show_evaluation_diagnostics,
    "debug-show-evaluations": _legacy.cmd_show_evaluations,
    "rebuild-strategy-adaptation-state": _legacy.cmd_rebuild_strategy_adaptation_state,
    "show-strategy-adaptation-states": _legacy.cmd_show_strategy_adaptation_states,
}
