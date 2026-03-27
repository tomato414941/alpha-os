from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "alpha_os_recovery"
ALPHA_PACKAGE = SRC_ROOT / "alpha"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
TESTS_ROOT = PROJECT_ROOT / "tests"
ALLOWED_ALPHA_MODULES = {
    "__init__.py",
}
CURRENT_HYPOTHESIS_ALIAS_PATHS = [
    SRC_ROOT / "cli.py",
    SRC_ROOT / "forward",
    SRC_ROOT / "hypotheses",
    SRC_ROOT / "paper" / "trader.py",
    SRC_ROOT / "validation",
    TESTS_ROOT / "test_forward.py",
    TESTS_ROOT / "test_hypotheses.py",
    TESTS_ROOT / "test_hypothesis_lifecycle.py",
    TESTS_ROOT / "test_paper.py",
    TESTS_ROOT / "test_trade_cli.py",
]
FORBIDDEN_TRACKER_ALIAS_ATTRS = {
    "register_alpha",
    "get_start_date",
    "get_realizable_returns",
    "tracked_alpha_ids",
    "save_alpha_signals",
    "get_alpha_signals",
    "get_alpha_signal_history",
}
FORBIDDEN_FORWARD_TYPE_NAMES = {
    "ForwardTracker",
    "ForwardRecord",
    "ForwardSummary",
}
FORBIDDEN_CURRENT_MODULES = {
    SRC_ROOT / "daemon" / "admission.py",
    SRC_ROOT / "daemon" / "alpha_generator.py",
    SRC_ROOT / "daemon" / "lifecycle.py",
    SRC_ROOT / "experiments" / "replay.py",
    SRC_ROOT / "paper" / "tactical.py",
    SRC_ROOT / "paper" / "simulator.py",
    SRC_ROOT / "pipeline" / "runner.py",
    SRC_ROOT / "research" / "pipeline_runner.py",
    SRC_ROOT / "research" / "replay_experiment.py",
    SRC_ROOT / "research" / "replay_simulator.py",
    SRC_ROOT / "research" / "deployment_planner.py",
    SRC_ROOT / "research" / "registry_signal_map.py",
}


def _forbidden_alpha_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "alpha_os_recovery.alpha" or alias.name.startswith("alpha_os_recovery.alpha."):
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                module = node.module or ""
                if module == "alpha_os_recovery.alpha" or module.startswith("alpha_os_recovery.alpha."):
                    violations.append(f"from {module} import ...")
            else:
                module = node.module or ""
                if module == "alpha" or module.startswith("alpha."):
                    violations.append(f"from {'.' * node.level}{module} import ...")

    return violations


def test_src_package_does_not_import_alpha_outside_compatibility_layer():
    violations: list[str] = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.is_relative_to(ALPHA_PACKAGE):
            continue
        for violation in _forbidden_alpha_imports(path):
            violations.append(f"{path.relative_to(PROJECT_ROOT)}: {violation}")

    assert violations == []


def test_scripts_and_tests_do_not_import_alpha_compatibility_package():
    violations: list[str] = []

    for root in (SCRIPTS_ROOT, TESTS_ROOT):
        for path in sorted(root.rglob("*.py")):
            if path.name == "test_architecture_boundaries.py":
                continue
            for violation in _forbidden_alpha_imports(path):
                violations.append(f"{path.relative_to(PROJECT_ROOT)}: {violation}")

    assert violations == []


def test_alpha_compatibility_package_contains_only_legacy_wrappers():
    actual_modules = {path.name for path in ALPHA_PACKAGE.glob("*.py")}

    assert actual_modules == ALLOWED_ALPHA_MODULES


def test_current_runtime_code_does_not_use_hypothesis_alpha_id_alias():
    violations: list[str] = []

    for target in CURRENT_HYPOTHESIS_ALIAS_PATHS:
        paths = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for path in paths:
            if "record.alpha_id" in path.read_text():
                violations.append(str(path.relative_to(PROJECT_ROOT)))

    assert violations == []


def test_repo_does_not_call_legacy_tracker_alias_methods():
    violations: list[str] = []

    for root in (SRC_ROOT, SCRIPTS_ROOT, TESTS_ROOT):
        for path in sorted(root.rglob("*.py")):
            if path == __file__:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_TRACKER_ALIAS_ATTRS:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}: {node.attr}")

    assert violations == []


def test_repo_does_not_use_legacy_forward_type_names():
    violations: list[str] = []

    for root in (SRC_ROOT, SCRIPTS_ROOT, TESTS_ROOT):
        for path in sorted(root.rglob("*.py")):
            if path in {
                SRC_ROOT / "forward" / "tracker.py",
                TESTS_ROOT / "test_forward.py",
                Path(__file__),
            }:
                continue
            text = path.read_text()
            for name in FORBIDDEN_FORWARD_TYPE_NAMES:
                if name in text:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}: {name}")

    assert violations == []


def test_legacy_runtime_modules_do_not_reappear_in_current_packages():
    assert [path for path in FORBIDDEN_CURRENT_MODULES if path.exists()] == []
