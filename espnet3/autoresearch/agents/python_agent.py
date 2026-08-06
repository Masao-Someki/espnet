"""Recipe-local Python agent loader."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def load_python_agent(target: str, recipe_dir: Path, agent_config):
    """Load a recipe-local agent class or factory."""
    recipe_dir = Path(recipe_dir).resolve()
    recipe_dir_str = str(recipe_dir)
    if recipe_dir_str not in sys.path:
        sys.path.insert(0, recipe_dir_str)

    module_path, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, attr_name)
    if isinstance(obj, type):
        return obj(config=agent_config, recipe_dir=recipe_dir)
    if callable(obj):
        return obj(config=agent_config, recipe_dir=recipe_dir)
    return obj
