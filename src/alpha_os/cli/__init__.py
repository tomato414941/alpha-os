from __future__ import annotations

import argparse
import sys
import types

from . import internal as _legacy
from .evaluation import COMMAND_HANDLERS as EVALUATION_COMMAND_HANDLERS
from .internal_commands import COMMAND_HANDLERS as INTERNAL_COMMAND_HANDLERS
from .research import COMMAND_HANDLERS as RESEARCH_COMMAND_HANDLERS
from .runtime import COMMAND_HANDLERS as RUNTIME_COMMAND_HANDLERS
from .runtime import register_runtime_parsers

CommandHandler = _legacy.CommandHandler if hasattr(_legacy, "CommandHandler") else object


def _get_subparsers(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise RuntimeError("CLI parser has no subparsers action")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = _legacy.build_cli_parser(include_runtime_parsers=False)
    register_runtime_parsers(_get_subparsers(parser))
    return parser


def build_command_handlers():
    handlers = {}
    for group in (
        RUNTIME_COMMAND_HANDLERS,
        EVALUATION_COMMAND_HANDLERS,
        RESEARCH_COMMAND_HANDLERS,
        INTERNAL_COMMAND_HANDLERS,
    ):
        overlap = handlers.keys() & group.keys()
        if overlap:
            raise ValueError(f"duplicate CLI command handlers: {sorted(overlap)}")
        handlers.update(group)
    return handlers


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    handlers = build_command_handlers()
    try:
        handler = handlers[args.command]
    except KeyError:
        parser.error(f"unknown command: {args.command}")
    try:
        return handler(args)
    except ValueError as exc:
        parser.error(str(exc))


def __getattr__(name: str):
    return getattr(_legacy, name)


class _CliModule(types.ModuleType):
    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        if hasattr(_legacy, name):
            setattr(_legacy, name, value)


sys.modules[__name__].__class__ = _CliModule
