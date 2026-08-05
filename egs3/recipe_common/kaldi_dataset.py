"""Dataset helpers for recipes backed by Kaldi data directories."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from espnet3.components.data.dataset_builder import DatasetBuilder


def _read_kaldi_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Kaldi file is missing: {path}")
    values = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                key, value = line.split(maxsplit=1)
                values[key] = value
    return values


def _read_audio(spec: str) -> np.ndarray:
    if spec.endswith("|"):
        process = subprocess.run(
            spec[:-1].strip(),
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        audio, _sample_rate = sf.read(io.BytesIO(process.stdout), dtype="float32")
        return np.asarray(audio, dtype=np.float32)
    audio, _sample_rate = sf.read(os.path.expandvars(spec), dtype="float32")
    return np.asarray(audio, dtype=np.float32)


class KaldiDataDataset(TorchDataset):
    """Read ``wav.scp`` and ``text`` from one Kaldi data directory."""

    split_aliases: dict[str, str] = {}

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
    ) -> None:
        recipe_root = Path(recipe_dir or Path.cwd()).resolve()
        data_root = Path(source_dir or recipe_root / "data").resolve()
        data_split = self.split_aliases.get(split, split)
        split_root = data_root / data_split
        self._wav = _read_kaldi_map(split_root / "wav.scp")
        self._text = _read_kaldi_map(split_root / "text")
        self._utt_ids = sorted(key for key in self._wav if key in self._text)
        if not self._utt_ids:
            raise RuntimeError(f"No paired speech/text entries found in {split_root}")

    def __len__(self) -> int:
        return len(self._utt_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        utt_id = self._utt_ids[int(index)]
        return {
            "speech": _read_audio(self._wav[utt_id]),
            "text": self._text[utt_id],
            "utt_id": utt_id,
        }


class KaldiDataBuilder(DatasetBuilder):
    """Validate an egs2-style Kaldi data directory."""

    required_splits: tuple[str, ...] = ("train", "dev", "test")
    split_aliases: dict[str, str] = {}
    source_env_var: str | None = None

    def resolve_root(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
    ) -> Path:
        candidates = [Path(source_dir)] if source_dir is not None else []
        if self.source_env_var and os.environ.get(self.source_env_var):
            candidates.append(Path(os.environ[self.source_env_var]))
        candidates.append(Path(recipe_dir) / "data")
        for candidate in candidates:
            if all(
                (candidate / self.split_aliases.get(split, split) / name).is_file()
                for split in self.required_splits
                for name in ("wav.scp", "text")
            ):
                return candidate
        locations = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Prepared Kaldi data not found for {type(self).__name__}; checked {locations}"
        )

    def is_source_prepared(self, recipe_dir, source_dir=None, **_kwargs):
        try:
            self.resolve_root(recipe_dir, source_dir)
        except FileNotFoundError:
            return False
        return True

    def prepare_source(self, recipe_dir, source_dir=None, **_kwargs):
        self.resolve_root(recipe_dir, source_dir)

    def is_built(self, recipe_dir, source_dir=None, **_kwargs):
        return self.is_source_prepared(recipe_dir, source_dir)

    def build(self, recipe_dir, source_dir=None, **_kwargs):
        self.prepare_source(recipe_dir, source_dir)
