"""Simple file lock utility."""

from __future__ import annotations

import contextlib
import fcntl
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an advisory file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
