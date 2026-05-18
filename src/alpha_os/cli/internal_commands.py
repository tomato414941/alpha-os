from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "init-db": _legacy.cmd_init_db,
    "init": _legacy.cmd_init_db,
    "register-signal-candidate": _legacy.cmd_register_signal,
    "deactivate-signal-candidate": _legacy.cmd_deactivate_signal,
    "activate-signal-candidate": _legacy.cmd_activate_signal,
    "decide-portfolio": _legacy.cmd_decide_portfolio,
    "debug-decide-portfolio-runtime": _legacy.cmd_debug_decide_portfolio_runtime,
    "debug-show-portfolio-decisions": _legacy.cmd_show_portfolio_decisions,
}
