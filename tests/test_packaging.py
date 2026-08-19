"""Keep declared distribution metadata consistent with what the package imports."""

import ast
import re
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "centerpoint"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

# Import name -> distribution name, for the third-party modules this package uses.
DISTRIBUTION_NAMES = {"numpy": "numpy", "torch": "torch"}


def load_pyproject():
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli", reason="no TOML parser available")
    return tomllib.loads(PYPROJECT_PATH.read_text())


def imported_top_level_modules():
    modules = set()
    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def third_party_imports():
    return {
        name
        for name in imported_top_level_modules()
        if name != "centerpoint" and name not in sys.stdlib_module_names
    }


def declared_requirement_names(project):
    names = set()
    for requirement in project.get("dependencies", []):
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        assert match is not None, f"unparsable requirement {requirement!r}"
        names.add(match.group(1).lower().replace("_", "-"))
    return names


def test_every_third_party_import_is_a_declared_dependency():
    project = load_pyproject()["project"]
    declared = declared_requirement_names(project)

    for module in sorted(third_party_imports()):
        assert module in DISTRIBUTION_NAMES, (
            f"{module!r} is imported by centerpoint but is not mapped to a distribution name"
        )
        assert DISTRIBUTION_NAMES[module] in declared, (
            f"{module!r} is imported by centerpoint but {DISTRIBUTION_NAMES[module]!r} "
            "is not declared in [project].dependencies"
        )


def test_declared_dependencies_are_actually_imported():
    project = load_pyproject()["project"]
    expected = {DISTRIBUTION_NAMES[module] for module in third_party_imports()}

    assert declared_requirement_names(project) == expected


def test_package_requires_python_matches_the_verified_interpreter_floor():
    project = load_pyproject()["project"]

    assert project["requires-python"] == ">=3.10"
