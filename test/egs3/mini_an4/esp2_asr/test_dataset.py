from pathlib import Path

from egs3.mini_an4.esp2_asr.dataset.builder import MiniAn4Builder
from egs3.mini_an4.esp2_asr.dataset.dataset import MiniAn4Dataset


def _write_manifest(recipe_dir: Path, rows: list[tuple[str, str, str]]) -> None:
    manifest_path = recipe_dir / "data" / "manifest" / "train.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(f"{uid}\t{wav}\t{text}" for uid, wav, text in rows)
    manifest_path.write_text(lines + "\n", encoding="utf-8")


def test_get_utt_id_reads_the_manifests_id_column(tmp_path, monkeypatch):
    """get_utt_id names an item from the manifest, without loading it."""
    monkeypatch.setattr(MiniAn4Builder, "is_source_prepared", lambda self, **_: True)
    monkeypatch.setattr(MiniAn4Builder, "is_built", lambda self, **_: True)
    _write_manifest(
        tmp_path,
        [
            ("an4_train_0", "missing_0.wav", "one"),
            ("an4_train_1", "missing_1.wav", "two"),
        ],
    )

    dataset = MiniAn4Dataset(split="train", recipe_dir=tmp_path)

    assert dataset.get_utt_id(0) == "an4_train_0"
    assert dataset.get_utt_id(1) == "an4_train_1"


def test_get_utt_id_does_not_read_the_audio_file(tmp_path, monkeypatch):
    """The manifest's wav path never has to exist for get_utt_id to work."""
    monkeypatch.setattr(MiniAn4Builder, "is_source_prepared", lambda self, **_: True)
    monkeypatch.setattr(MiniAn4Builder, "is_built", lambda self, **_: True)
    _write_manifest(tmp_path, [("an4_train_0", "does_not_exist.wav", "one")])

    dataset = MiniAn4Dataset(split="train", recipe_dir=tmp_path)

    assert not (tmp_path / "data" / "does_not_exist.wav").exists()
    assert dataset.get_utt_id(0) == "an4_train_0"


def test_getitem_does_not_carry_an_utt_id_field(tmp_path, monkeypatch):
    """The item itself has no id; the dataset is the only source of one."""
    monkeypatch.setattr(MiniAn4Builder, "is_source_prepared", lambda self, **_: True)
    monkeypatch.setattr(MiniAn4Builder, "is_built", lambda self, **_: True)

    import numpy as np
    import soundfile as sf

    manifest_dir = tmp_path / "data"
    wav_path = manifest_dir / "utt0.wav"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, np.zeros(1600, dtype=np.float32), 16000)
    _write_manifest(tmp_path, [("an4_train_0", str(wav_path), "one")])

    dataset = MiniAn4Dataset(split="train", recipe_dir=tmp_path)

    assert set(dataset[0]) == {"speech", "text"}
