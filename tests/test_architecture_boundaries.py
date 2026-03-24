from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "alpha_os"
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


def _forbidden_alpha_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "alpha_os.alpha" or alias.name.startswith("alpha_os.alpha."):
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                module = node.module or ""
                if module == "alpha_os.alpha" or module.startswith("alpha_os.alpha."):
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
