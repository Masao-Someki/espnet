"""Dynamic stage loader."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from espnet3.autoresearch.core.errors import StageLoadError


def load_stage_class(target: str, recipe_dir: Path | None = None) -> type:
    """Load a stage class by dotted import path."""
    if recipe_dir is not None:
        recipe_dir_str = str(recipe_dir.resolve())
        if recipe_dir_str not in sys.path:
            sys.path.insert(0, recipe_dir_str)

    module_path, class_name = target.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        recipe_hint = ""
        if recipe_dir is not None:
            recipe_hint = (
                f"\nIf this is recipe-local, verify recipe_dir='{recipe_dir}' and "
                f"module file '{recipe_dir / module_path.replace('.', '/')}.py'."
            )
        raise StageLoadError(
            f"Cannot import module '{module_path}' for stage target '{target}'."
            f"{recipe_hint}\nOriginal error: {exc}"
        ) from exc

    if not hasattr(module, class_name):
        raise StageLoadError(f"Module '{module_path}' has no class '{class_name}'.")
    return getattr(module, class_name)
