from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "init-db": _legacy.cmd_init_db,
    "register-signal-candidate": _legacy.cmd_register_signal,
    "deactivate-signal-candidate": _legacy.cmd_deactivate_signal,
    "activate-signal-candidate": _legacy.cmd_activate_signal,
    "decide-portfolio": _legacy.cmd_decide_portfolio,
    "debug-decide-portfolio-runtime": _legacy.cmd_debug_decide_portfolio_runtime,
    "debug-show-portfolio-decisions": _legacy.cmd_show_portfolio_decisions,
    "debug-validate-subject-set": _legacy.cmd_validate_subject_set,
    "validate-strategy": _legacy.cmd_validate_strategy,
    "debug-write-validation-spec": _legacy.cmd_debug_write_validation_spec,
    "debug-run-validation": _legacy.cmd_debug_run_validation,
    "debug-show-validation": _legacy.cmd_debug_show_validation,
    "debug-summarize-validation": _legacy.cmd_debug_summarize_validation,
}

