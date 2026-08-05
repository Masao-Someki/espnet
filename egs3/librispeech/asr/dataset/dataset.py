"""Raw-directory LibriSpeech dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.librispeech.asr.dataset.builder import resolve_source_root
from egs3.librispeech.asr.dataset.builder import _cache_options, _cache_root
from egs3.librispeech.asr.dataset.builder import _cache_manifest_path
from egs3.librispeech_100.asr.dataset.dataset import _scan_split


class _OmniIOReader:
    """Read one omniio-compatible Parquet manifest lazily by utterance ID."""

    def __init__(self, manifest_path: Path) -> None:
        try:
            import duckdb
            from omniio.interface import audio_read
        except ImportError as exc:
            raise ImportError(
                "omniio cache loading requires omniio, duckdb, and pyarrow."
            ) from exc

        self._audio_read = audio_read
        connection = duckdb.connect()
        self._table = connection.execute(
            f"SELECT * FROM read_parquet('{manifest_path}')"
        ).pl()
        connection.close()
        self._index = {
            utt_id: index
            for index, utt_id in enumerate(self._table["utt_id"].to_physical())
        }

    def __getitem__(self, utt_id: str) -> tuple[np.ndarray, int]:
        row = self._table.row(self._index[utt_id], named=True)
        data = self._audio_read(
            row["path"],
            start_offset=row["start_byte_offset"],
            file_size=row["file_size_bytes"],
        )
        return np.asarray(data.array.T), int(data.sample_rate)


class LibriSpeechDataset(TorchDataset):
    """Read any supported LibriSpeech split from the original directory layout."""

    _SPLITS = {
        "train-clean-100", "train-clean-360", "train-other-500",
        "dev-clean", "dev-other", "test-clean", "test-other",
    }

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
        cache: dict[str, Any] | None = None,
    ) -> None:
        if split not in self._SPLITS:
            raise ValueError(f"Unknown LibriSpeech split: {split}")
        recipe_root = Path(
            recipe_dir or Path(__file__).resolve().parents[1]
        ).resolve()
        root = resolve_source_root(recipe_root, source_dir)
        self._examples = _scan_split(root / split)
        options = _cache_options(cache)
        self._omniio = None
        if options.get("enabled", False):
            if options.get("backend", "omniio") != "omniio":
                raise ValueError("Only the omniio cache backend is supported")
            manifest = _cache_manifest_path(
                _cache_root(recipe_root, options["cache_dir"]), split
            )
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"omniio cache is missing: {manifest}. "
                    "Run the create_dataset stage with cache.enabled=true first."
                )
            self._omniio = _OmniIOReader(manifest)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self._examples[int(idx)]
        if self._omniio is None:
            audio, _sample_rate = sf.read(str(example.audio_path))
        else:
            audio, _sample_rate = self._omniio[example.utt_id]
            if audio.ndim == 2 and audio.shape[-1] == 1:
                audio = audio[:, 0]
        return {
            "speech": np.asarray(audio, dtype=np.float32),
            "text": example.text,
            "utt_id": example.utt_id,
        }
