from __future__ import annotations

import argparse
from collections.abc import Callable

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "debug-register-signal-candidate-spec": _legacy.cmd_register_signal_spec,
    "debug-show-signal-candidate-specs": _legacy.cmd_show_signal_specs,
    "debug-register-observable": _legacy.cmd_register_observable,
    "debug-show-observables": _legacy.cmd_show_observables,
    "debug-backfill-subject-set": _legacy.cmd_backfill_subject_set,
    "debug-backfill-signal-discovery": _legacy.cmd_backfill_signal_discovery,
    "inspect-subject-set": _legacy.cmd_inspect_subject_set,
    "debug-show-meta-predictions": _legacy.cmd_show_meta_predictions,
    "debug-compare-meta-aggregations": _legacy.cmd_compare_meta_aggregations,
    "debug-register-subject-set": _legacy.cmd_register_subject_set,
    "debug-show-subject-sets": _legacy.cmd_show_subject_sets,
    "check-subject-set-backend": _legacy.cmd_check_subject_set_backend,
    "debug-screen-signal-discovery": _legacy.cmd_screen_signal_discovery,
    "debug-compress-screening-result": _legacy.cmd_compress_screening_result,
}
