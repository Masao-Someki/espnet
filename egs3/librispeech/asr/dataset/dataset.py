"""Raw-directory LibriSpeech dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.librispeech.asr.dataset.builder import resolve_source_root
from egs3.librispeech_100.asr.dataset.dataset import _scan_split


class LibriSpeechDataset(TorchDataset):
    """Read any supported LibriSpeech split from the original directory layout."""

    _SPLITS = {
        "train-clean-100", "train-clean-360", "train-other-500",
        "dev-clean", "dev-other", "test-clean", "test-other",
    }

    def __init__(self, split: str, recipe_dir: str | Path | None = None,
                 source_dir: str | Path | None = None) -> None:
        if split not in self._SPLITS:
            raise ValueError(f"Unknown LibriSpeech split: {split}")
        recipe_root = Path(recipe_dir or Path(__file__).resolve().parents[1]).resolve()
        root = resolve_source_root(recipe_root, source_dir)
        self._examples = _scan_split(root / split)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self._examples[int(idx)]
        audio, _sample_rate = sf.read(str(example.audio_path))
        return {"speech": np.asarray(audio, dtype=np.float32), "text": example.text,
                "utt_id": example.utt_id}
