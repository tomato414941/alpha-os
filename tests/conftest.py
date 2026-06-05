"""Shared test helpers for alpha-os."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.current)


def load_example_module(path: str) -> ModuleType:
    example_path = Path(path)
    module_name = example_path.stem
    spec = importlib.util.spec_from_file_location(module_name, example_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
