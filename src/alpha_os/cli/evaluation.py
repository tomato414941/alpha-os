from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "run-evaluation": _legacy.cmd_run_evaluation,
    "run-walk-forward": _legacy.cmd_run_walk_forward_evaluation,
}
