"""Tokenizer helpers for Kaldi data-directory recipes."""

from pathlib import Path

from egs3.recipe_common.kaldi_dataset import _read_kaldi_map


def gather_training_text(recipe_dir=None, source_dir=None, split="train"):
    """Collect text from a recipe's Kaldi training directory."""
    root = Path(source_dir or Path(recipe_dir or Path.cwd()) / "data")
    return list(_read_kaldi_map(root / split / "text").values())
