from __future__ import annotations

import subprocess
import sys

import pytest

from alpha_os.cli import build_cli_parser, build_command_handlers
from alpha_os.cli.evaluation import COMMAND_HANDLERS as EVALUATION_COMMAND_HANDLERS
from alpha_os.cli.internal_commands import COMMAND_HANDLERS as INTERNAL_COMMAND_HANDLERS
from alpha_os.cli.research import COMMAND_HANDLERS as RESEARCH_COMMAND_HANDLERS
from alpha_os.cli.runtime import COMMAND_HANDLERS as RUNTIME_COMMAND_HANDLERS


def _parser_command_names() -> set[str]:
    parser = build_cli_parser()
    for action in parser._actions:
        if getattr(action, "choices", None):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


def test_cli_dispatch_is_grouped_by_use_case_module():
    grouped_handlers = (
        RUNTIME_COMMAND_HANDLERS
        | EVALUATION_COMMAND_HANDLERS
        | RESEARCH_COMMAND_HANDLERS
        | INTERNAL_COMMAND_HANDLERS
    )

    assert set(build_command_handlers()) == set(grouped_handlers)
    assert _parser_command_names() == set(grouped_handlers)


def test_cli_entrypoint_keeps_public_help_surface(capsys):
    parser = build_cli_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])

    captured = capsys.readouterr().out
    assert "apply-runtime-manifest" in captured
    assert "run-walk-forward-evaluation" in captured
    assert "debug-apply-evaluation" not in captured


def test_cli_entrypoint_preserves_value_error_exit_contract(monkeypatch):
    import alpha_os.cli as cli

    def fail(_args):
        raise ValueError("bad command input")

    monkeypatch.setattr(cli, "build_command_handlers", lambda: {"list-runtime-manifests": fail})

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["list-runtime-manifests"])

    assert exc_info.value.code == 2


def test_cli_package_module_entrypoint_preserves_help():
    result = subprocess.run(
        [sys.executable, "-m", "alpha_os.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "apply-runtime-manifest" in result.stdout
    assert "debug-apply-evaluation" not in result.stdout
