from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from . import internal as _legacy

CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True)
class CliCommand:
    name: str
    handler: CommandHandler
    visibility: str
    register_parser: Callable[[argparse._SubParsersAction], None]


def _hide_subparser_help(
    sub: argparse._SubParsersAction,
    name: str,
) -> None:
    sub._choices_actions = [
        action for action in sub._choices_actions if getattr(action, "dest", None) != name
    ]


def _register_apply_manifest(
    sub: argparse._SubParsersAction,
    name: str,
) -> None:
    parser = sub.add_parser(
        name,
        help=(
            "Apply runtime manifest resources including observables, signal specs, "
            "subject sets, strategy specs, and evaluation specs"
        ),
    )
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--manifest", type=str, required=True)


def _register_list_manifests(
    sub: argparse._SubParsersAction,
    name: str,
) -> None:
    sub.add_parser(
        name,
        help="List checked-in runtime manifests with categories",
    )


COMMANDS: tuple[CliCommand, ...] = (
    CliCommand(
        name="apply-manifest",
        handler=_legacy.cmd_apply_runtime_manifest,
        visibility="public",
        register_parser=lambda sub: _register_apply_manifest(
            sub,
            "apply-manifest",
        ),
    ),
    CliCommand(
        name="list-manifests",
        handler=_legacy.cmd_list_runtime_manifests,
        visibility="public",
        register_parser=lambda sub: _register_list_manifests(
            sub,
            "list-manifests",
        ),
    ),
)

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    command.name: command.handler for command in COMMANDS
}


def register_runtime_parsers(sub: argparse._SubParsersAction) -> None:
    for command in COMMANDS:
        command.register_parser(sub)
        if command.visibility != "public":
            _hide_subparser_help(sub, command.name)
