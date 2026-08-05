"""Source validation for the full LibriSpeech corpus."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _config() -> dict:
    resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(resource) as path:
        return load_config_with_defaults(str(path), resolve=False)["builder"]


_CFG = _config()


def resolve_source_root(recipe_dir: str | Path, source_dir: str | Path | None = None) -> Path:
    """Return the LibriSpeech root from the recipe, override, or environment."""
    candidates = [Path(recipe_dir) / _CFG["dataset_path"]]
    if source_dir is not None:
        candidates.append(Path(source_dir))
    if os.environ.get(_CFG["source_env_var"]):
        candidates.append(Path(os.environ[_CFG["source_env_var"]]))
    for candidate in candidates:
        root = candidate / "LibriSpeech" if (candidate / "LibriSpeech").is_dir() else candidate
        if root.is_dir() and all((root / split).is_dir() for split in _CFG["required_splits"]):
            return root
    raise FileNotFoundError("Full LibriSpeech source is missing required splits")


class LibriSpeechBuilder(DatasetBuilder):
    """Validate the raw full LibriSpeech directory."""

    def is_source_prepared(self, recipe_dir, source_dir=None, **_kwargs):
        try:
            resolve_source_root(recipe_dir, source_dir)
        except FileNotFoundError:
            return False
        return True

    def prepare_source(self, recipe_dir, source_dir=None, **_kwargs):
        resolve_source_root(recipe_dir, source_dir)

    def is_built(self, recipe_dir, source_dir=None, **_kwargs):
        return self.is_source_prepared(recipe_dir, source_dir)

    def build(self, recipe_dir, source_dir=None, **_kwargs):
        self.prepare_source(recipe_dir, source_dir)
