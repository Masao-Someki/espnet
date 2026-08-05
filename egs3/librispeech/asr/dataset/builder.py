"""Source validation for the full LibriSpeech corpus."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

from egs3.librispeech_100.asr.dataset.dataset import _scan_split


def _config() -> dict:
    resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(resource) as path:
        return load_config_with_defaults(str(path), resolve=False)["builder"]


_CFG = _config()


def _cache_config() -> dict[str, Any]:
    resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(resource) as path:
        return load_config_with_defaults(str(path), resolve=False).get("cache", {})


_CACHE_CFG = _cache_config()


def _cache_options(cache: dict[str, Any] | None) -> dict[str, Any]:
    options = dict(_CACHE_CFG)
    if cache is not None:
        options.update(dict(cache))
    return options


def _cache_root(recipe_dir: str | Path, cache_dir: str | Path) -> Path:
    path = Path(cache_dir)
    return path if path.is_absolute() else Path(recipe_dir) / path


def _cache_manifest_path(cache_root: Path, split: str) -> Path:
    return cache_root / f"{split}.parquet"


def _build_omniio_cache(
    source_root: Path,
    cache_root: Path,
    splits: list[str],
) -> None:
    """Materialize LibriSpeech audio and an omniio-compatible index."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "omniio cache creation requires pyarrow; install omniio and pyarrow."
        ) from exc

    cache_root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        split_cache = cache_root / split
        split_cache.mkdir(parents=True, exist_ok=True)
        rows = []
        for example in _scan_split(source_root / split):
            output_path = split_cache / f"{example.utt_id}.wav"
            if not output_path.is_file():
                audio, sample_rate = sf.read(str(example.audio_path), dtype="float32")
                sf.write(str(output_path), np.asarray(audio), sample_rate)
            rows.append(
                {
                    "utt_id": example.utt_id,
                    "path": str(output_path.resolve()),
                    "start_byte_offset": 0,
                    "file_size_bytes": output_path.stat().st_size,
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), _cache_manifest_path(cache_root, split))


def _cache_is_built(cache_root: Path, splits: list[str]) -> bool:
    return all(_cache_manifest_path(cache_root, split).is_file() for split in splits)


def resolve_source_root(
    recipe_dir: str | Path, source_dir: str | Path | None = None
) -> Path:
    """Return the LibriSpeech root from the recipe, override, or environment."""
    candidates = [Path(recipe_dir) / _CFG["dataset_path"]]
    if source_dir is not None:
        candidates.append(Path(source_dir))
    if os.environ.get(_CFG["source_env_var"]):
        candidates.append(Path(os.environ[_CFG["source_env_var"]]))
    for candidate in candidates:
        root = (
            candidate / "LibriSpeech"
            if (candidate / "LibriSpeech").is_dir()
            else candidate
        )
        if root.is_dir() and all(
            (root / split).is_dir() for split in _CFG["required_splits"]
        ):
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

    def is_built(self, recipe_dir, source_dir=None, cache=None, **_kwargs):
        if not self.is_source_prepared(recipe_dir, source_dir):
            return False
        options = _cache_options(cache)
        if not options.get("enabled", False):
            return True
        if options.get("backend", "omniio") != "omniio":
            raise ValueError("Only the omniio cache backend is supported")
        cache_root = _cache_root(recipe_dir, options["cache_dir"])
        return _cache_is_built(cache_root, _CFG["required_splits"])

    def build(self, recipe_dir, source_dir=None, cache=None, **_kwargs):
        self.prepare_source(recipe_dir, source_dir)
        options = _cache_options(cache)
        if not options.get("enabled", False):
            return
        if options.get("backend", "omniio") != "omniio":
            raise ValueError("Only the omniio cache backend is supported")
        cache_root = _cache_root(recipe_dir, options["cache_dir"])
        _build_omniio_cache(
            resolve_source_root(recipe_dir, source_dir),
            cache_root,
            _CFG["required_splits"],
        )
