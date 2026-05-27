from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "init": _legacy.cmd_init_db,
}
