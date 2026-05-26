from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "init-db": _legacy.cmd_init_db,
    "init": _legacy.cmd_init_db,
    "decide-portfolio": _legacy.cmd_decide_portfolio,
    "debug-show-portfolio-decisions": _legacy.cmd_show_portfolio_decisions,
}
