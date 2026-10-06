"""The declared item fields match what LibriSpeech100Dataset actually returns."""

from pathlib import Path

import numpy as np
import soundfile as sf

from egs3.librispeech_100.esp2_asr.dataset.dataset import LibriSpeech100Dataset
from espnet3.components.data.contract import check_item

_ALL_SPLITS = (
    "train-clean-100",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
)


def _write_fake_corpus(root: Path) -> None:
    """Write a minimal fake LibriSpeech tree under ``root/download/LibriSpeech``."""
    librispeech_root = root / "download" / "LibriSpeech"
    for split in _ALL_SPLITS:
        (librispeech_root / split).mkdir(parents=True, exist_ok=True)

    speaker_dir = librispeech_root / "train-clean-100" / "19" / "198"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    utt_id = "19-198-0000"
    (speaker_dir / "19-198.trans.txt").write_text(
        f"{utt_id} HELLO WORLD\n", encoding="utf-8"
    )
    sf.write(
        str(speaker_dir / f"{utt_id}.flac"),
        np.zeros(1600, dtype=np.float32),
        16000,
    )


def test_fields_match_a_real_item(tmp_path: Path) -> None:
    _write_fake_corpus(tmp_path)

    dataset = LibriSpeech100Dataset(split="train-clean-100", recipe_dir=tmp_path)
    item = dataset[0]

    assert sorted(item) == ["speech", "text"]
    check_item(LibriSpeech100Dataset.fields, item, "LibriSpeech100Dataset[0]")


def test_fields_declares_speech_audio_and_text_text() -> None:
    kinds = {f.name: f.kind for f in LibriSpeech100Dataset.fields}
    assert kinds == {"speech": "audio", "text": "text"}
