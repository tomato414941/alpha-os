from __future__ import annotations

import subprocess
import sys

import pytest

from alpha_os.cli import build_cli_parser, build_command_handlers
from alpha_os.cli.evaluation import COMMAND_HANDLERS as EVALUATION_COMMAND_HANDLERS
from alpha_os.cli.internal_commands import COMMAND_HANDLERS as INTERNAL_COMMAND_HANDLERS
from alpha_os.cli.research import COMMAND_HANDLERS as RESEARCH_COMMAND_HANDLERS
from alpha_os.cli.runtime import COMMANDS as RUNTIME_COMMANDS
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


def test_runtime_commands_own_public_parser_registration():
    assert {
        command.name
        for command in RUNTIME_COMMANDS
        if command.visibility == "public"
    } == {
        "apply-manifest",
        "list-manifests",
    }

    parser = build_cli_parser()
    args = parser.parse_args(
        [
            "run-diagnostic-evaluation",
            "--manifest",
            "fixture_daily_diagnostic",
            "--evaluation-spec-id",
            "fixture_daily_diagnostic_eval",
            "--dry-run",
            "--check",
        ]
    )

    assert args.command == "run-diagnostic-evaluation"
    assert args.manifest == "fixture_daily_diagnostic"
    assert args.evaluation_spec_id == "fixture_daily_diagnostic_eval"
    assert args.dry_run is True
    assert args.check is True


def test_cli_help_surface_is_fixed_to_golden_path_commands(capsys):
    parser = build_cli_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])

    captured = capsys.readouterr().out
    public_commands = {
        "init",
        "apply-manifest",
        "list-manifests",
        "run-evaluation",
        "run-walk-forward",
        "show-report",
        "show-diagnostics",
    }
    hidden_commands = {
        "apply-runtime-manifest",
        "list-runtime-manifests",
        "run-walk-forward-evaluation",
        "show-evaluation-report",
        "show-evaluation-diagnostics",
        "run-diagnostic-evaluation",
        "inspect-runtime-resources",
        "debug-status",
        "debug-apply-evaluation",
        "run-signal-discovery",
    }

    for command in public_commands:
        assert command in captured
    for command in hidden_commands:
        assert command not in captured


def test_cli_entrypoint_keeps_public_help_surface(capsys):
    parser = build_cli_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])

    captured = capsys.readouterr().out
    assert "apply-manifest" in captured
    assert "run-walk-forward" in captured
    assert "show-report" in captured
    assert "show-diagnostics" in captured
    assert "apply-runtime-manifest" not in captured
    assert "run-walk-forward-evaluation" not in captured
    assert "run-diagnostic-evaluation" not in captured
    assert "debug-apply-evaluation" not in captured


def test_cli_public_aliases_parse_to_existing_arguments():
    parser = build_cli_parser()

    init_args = parser.parse_args(["init", "--db", "runtime.db"])
    assert init_args.command == "init"
    assert init_args.db == "runtime.db"

    apply_args = parser.parse_args(
        [
            "apply-manifest",
            "--manifest",
            "examples/minimal_oos.json",
            "--db",
            "runtime.db",
        ]
    )
    assert apply_args.command == "apply-manifest"
    assert apply_args.manifest == "examples/minimal_oos.json"
    assert apply_args.db == "runtime.db"

    run_args = parser.parse_args(
        [
            "run-walk-forward",
            "--evaluation-spec-id",
            "minimal_oos_eval",
            "--db",
            "runtime.db",
        ]
    )
    assert run_args.command == "run-walk-forward"
    assert run_args.evaluation_spec_id == "minimal_oos_eval"
    assert run_args.db == "runtime.db"

    report_args = parser.parse_args(["show-report", "--db", "runtime.db"])
    assert report_args.command == "show-report"
    assert report_args.db == "runtime.db"

    diagnostics_args = parser.parse_args(["show-diagnostics", "--db", "runtime.db"])
    assert diagnostics_args.command == "show-diagnostics"
    assert diagnostics_args.db == "runtime.db"


def test_cli_legacy_public_commands_remain_parseable():
    parser = build_cli_parser()

    apply_args = parser.parse_args(
        [
            "apply-runtime-manifest",
            "--manifest",
            "examples/minimal_oos.json",
            "--db",
            "runtime.db",
        ]
    )
    assert apply_args.command == "apply-runtime-manifest"
    assert apply_args.manifest == "examples/minimal_oos.json"

    list_args = parser.parse_args(["list-runtime-manifests"])
    assert list_args.command == "list-runtime-manifests"

    run_args = parser.parse_args(
        [
            "run-walk-forward-evaluation",
            "--evaluation-spec-id",
            "minimal_oos_eval",
            "--db",
            "runtime.db",
        ]
    )
    assert run_args.command == "run-walk-forward-evaluation"
    assert run_args.evaluation_spec_id == "minimal_oos_eval"

    report_args = parser.parse_args(["show-evaluation-report", "--db", "runtime.db"])
    assert report_args.command == "show-evaluation-report"

    diagnostics_args = parser.parse_args(
        ["show-evaluation-diagnostics", "--db", "runtime.db"]
    )
    assert diagnostics_args.command == "show-evaluation-diagnostics"


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
    assert "apply-manifest" in result.stdout
    assert "run-walk-forward" in result.stdout
    assert "apply-runtime-manifest" not in result.stdout
    assert "run-walk-forward-evaluation" not in result.stdout
    assert "run-diagnostic-evaluation" not in result.stdout
    assert "debug-apply-evaluation" not in result.stdout
