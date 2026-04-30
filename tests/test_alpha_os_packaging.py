from __future__ import annotations

import tomllib
from pathlib import Path


def _project_metadata() -> dict[str, object]:
    with Path("pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]


def test_project_metadata_exposes_console_script_and_repository_urls():
    project = _project_metadata()

    assert project["readme"] == "README.md"
    assert project["scripts"] == {"alpha-os": "alpha_os.cli:main"}
    assert project["urls"] == {
        "Repository": "https://github.com/tomato414941/alpha-os",
        "Issues": "https://github.com/tomato414941/alpha-os/issues",
    }


def test_heavy_research_and_exchange_dependencies_are_optional():
    project = _project_metadata()

    dependencies = set(project["dependencies"])
    assert "deap>=1.4" not in dependencies
    assert "ribs[all]>=0.7" not in dependencies
    assert "ccxt>=4.0" not in dependencies

    optional = project["optional-dependencies"]
    assert optional["research"] == ["deap>=1.4", "ribs[all]>=0.7"]
    assert optional["exchange"] == ["ccxt>=4.0"]
    assert optional["optimizer"] == ["cvxpy>=1.6", "skfolio==0.16.1"]
