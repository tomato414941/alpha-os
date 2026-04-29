from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "apply-runtime-manifest": _legacy.cmd_apply_runtime_manifest,
    "run-diagnostic-evaluation": _legacy.cmd_run_diagnostic_evaluation,
    "list-runtime-manifests": _legacy.cmd_list_runtime_manifests,
    "inspect-runtime-resources": _legacy.cmd_inspect_runtime_resources,
    "debug-status": _legacy.cmd_status,
}

