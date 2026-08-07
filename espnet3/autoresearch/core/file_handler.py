"""Locked file I/O helpers for AutoResearch."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from espnet3.autoresearch.core.locking import file_lock


class LockedFileHandler:
    """File I/O operations with advisory locking."""

    def atomic_write_text(self, path: Path, text: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def append_text(self, path: Path, text: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with file_lock(lock_path):
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)

    def named_lock(self, base_dir: Path, name: str):
        lock_path = Path(base_dir) / f".lock_{name}"
        return file_lock(lock_path)
