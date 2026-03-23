from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "alpha_os"
ALPHA_PACKAGE = SRC_ROOT / "alpha"


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
