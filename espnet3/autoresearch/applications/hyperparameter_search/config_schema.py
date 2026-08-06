"""Config helpers for AutoResearch hyperparameter search."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.autoresearch.core.serialization import save_yaml
from espnet3.utils.config_utils import load_and_merge_config


def load_autoresearch_config(recipe_dir: Path, config_path: Path):
    """Load the recipe autoresearch config."""
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    return cfg


def load_recipe_stage_config(
    recipe_dir: Path, config_path: Path, config_name: str, resolve: bool = True
):
    """Load a recipe config merged with TEMPLATE defaults."""
    return load_and_merge_config(
        recipe_dir / config_path,
        config_name=config_name,
        default_package="egs3.TEMPLATE.asr",
        resolve=resolve,
    )


def write_resolved_config(path: Path, config) -> None:
    """Write a resolved config to disk."""
    save_yaml(path, OmegaConf.to_container(config, resolve=True))
