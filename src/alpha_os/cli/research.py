from __future__ import annotations

import argparse
from collections.abc import Callable


CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {}
