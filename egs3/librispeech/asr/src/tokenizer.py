"""Tokenizer text collection for full LibriSpeech."""

from pathlib import Path

from egs3.librispeech.asr.dataset.builder import resolve_source_root
from egs3.librispeech_100.asr.src.tokenizer import _parse_transcript_file


def gather_training_text(recipe_dir=None, source_dir=None):
    """Collect transcripts from all LibriSpeech training splits."""
    root = resolve_source_root(Path(recipe_dir).resolve(), source_dir)
    texts = []
    for split in ("train-clean-100", "train-clean-360", "train-other-500"):
        for path in (root / split).rglob("*.trans.txt"):
            texts.extend(_parse_transcript_file(path))
    return texts
