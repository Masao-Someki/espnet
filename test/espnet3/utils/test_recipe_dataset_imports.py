"""Static checks that recipe dataset modules under egs3/ use absolute imports."""

import ast
from pathlib import Path
from typing import List, Tuple

EGS3_ROOT = Path(__file__).resolve().parents[3] / "egs3"


def find_relative_imports_in_recipe_datasets(egs3_root: Path) -> List[Tuple[Path, int]]:
    """Return every relative import found in egs3/**/dataset/**/*.py.

    Each recipe's ``dataset/__init__.py`` (and any module under its
    ``dataset/`` package) is parsed with ``ast.parse`` only -- the recipe is
    never imported -- so this check does not depend on whichever optional
    recipe dependencies happen to be installed.

    Args:
        egs3_root: Path to the ``egs3/`` directory to scan.

    Returns:
        A list of ``(file_path, line_number)`` tuples, one per ``ImportFrom``
        node whose ``level`` is 1 or more (a relative import, e.g.
        ``from .builder import ...`` or ``from ..dataset import ...``).
    """
    violations: List[Tuple[Path, int]] = []
    for path in sorted(egs3_root.glob("**/dataset/**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 1:
                violations.append((path, node.lineno))
    return violations


def test_recipe_dataset_modules_use_absolute_imports():
    """egs3/**/dataset/**/*.py must not use relative imports.

    A relative import (``from .builder import Foo``) makes the imported
    class's fully-qualified path depend on how ``espnet3`` loaded the
    recipe's ``dataset/__init__.py``: when ``data_src`` is omitted, the
    module is loaded from an absolute-path-derived synthetic module name, so
    a relatively-imported class ends up nested under that synthetic name and
    its class path changes with the checkout location. The espnet3 recipe
    convention (see ``egs3/CLAUDE.md``) is to always address dataset modules
    absolutely, e.g. ``from egs3.<dataset>.<task>.dataset.builder import
    Foo``, which resolves to a stable class path regardless of checkout
    location.
    """
    violations = find_relative_imports_in_recipe_datasets(EGS3_ROOT)
    assert (
        not violations
    ), "Relative imports found in recipe dataset modules:\n" + "\n".join(
        f"  {path}:{lineno}" for path, lineno in violations
    )
