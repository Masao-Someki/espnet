"""Confirm the spgispeech recipe's declared item fields match real samples."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from egs3.spgispeech.esp2_asr.dataset.builder import SPGISpeechBuilder
from egs3.spgispeech.esp2_asr.dataset.dataset import SPGISpeechDataset
from espnet3.api.inference import Field
from espnet3.components.contract.dataset import check_item


def _write_fake_corpus(source_root: Path) -> None:
    """Write a minimal train.csv plus matching audio under ``source_root``.

    Mirrors the real SPGISpeech layout: a ``|``-delimited manifest with a
    header row, and ``spgispeech/train/<hash>/<n>.wav`` audio files.
    """
    audio_dir = source_root / "spgispeech" / "train" / "0000"
    audio_dir.mkdir(parents=True)
    sf.write(audio_dir / "1.wav", np.zeros(1600, dtype=np.float32), 16000)
    sf.write(audio_dir / "2.wav", np.zeros(1600, dtype=np.float32), 16000)

    with (source_root / "train.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["wav_filename", "wav_filesize", "transcript"])
        writer.writerow(["0000/1.wav", "3200", "Hello, World!"])
        writer.writerow(["0000/2.wav", "3200", "Another Sentence."])

    (source_root / "val.csv").write_text(
        "wav_filename|wav_filesize|transcript\n", encoding="utf-8"
    )
    (source_root / "spgispeech" / "val").mkdir(parents=True)


def test_dataset_fields_match_real_items(tmp_path):
    """Every item SPGISpeechDataset yields must satisfy its declared fields."""
    _write_fake_corpus(tmp_path)

    dataset = SPGISpeechDataset(split="train", recipe_dir=tmp_path, source_dir=tmp_path)
    assert len(dataset) == 2

    for idx in range(len(dataset)):
        check_item(dataset.fields, dataset[idx], where=f"SPGISpeechDataset[{idx}]")


def test_dataset_fields_declaration():
    """The declared fields must name exactly the keys __getitem__ returns."""
    assert SPGISpeechDataset.fields == (Field("speech", "audio"), Field("text", "text"))


def test_builder_has_no_manifest_columns():
    """The builder intentionally omits manifest_columns (see its docstring)."""
    assert SPGISpeechBuilder.manifest_columns is None
    assert SPGISpeechBuilder().built_manifests() == {}
