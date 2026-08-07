"""Serialization helpers for AutoResearch."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from espnet3.autoresearch.core.file_handler import LockedFileHandler

_FILE_HANDLER = LockedFileHandler()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_plain_data(value: Any) -> Any:
    """Convert nested dataclasses/configs/paths to plain JSON-safe values."""
    if is_dataclass(value):
        return {k: to_plain_data(v) for k, v in asdict(value).items()}
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(v) for v in value]
    return value


def dumps_json(data: Any, indent: int = 2) -> str:
    """Serialize data to JSON."""
    return json.dumps(to_plain_data(data), ensure_ascii=False, indent=indent)


def loads_json(text: str) -> Any:
    """Parse JSON text."""
    return json.loads(text)


def load_yaml(path: Path) -> Any:
    """Load YAML via OmegaConf and return plain Python data."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def save_yaml(path: Path, data: Any, resolve: bool = True) -> None:
    """Atomically write YAML data."""
    plain = to_plain_data(data)
    cfg = OmegaConf.create(plain)
    yaml_text = OmegaConf.to_yaml(cfg, resolve=resolve)
    _FILE_HANDLER.atomic_write_text(path, yaml_text)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text."""
    _FILE_HANDLER.atomic_write_text(path, text)


def append_text(path: Path, text: str) -> None:
    """Append text with a sibling lock file."""
    _FILE_HANDLER.append_text(path, text)


def named_lock(base_dir: Path, name: str):
    """Acquire a named lock under a base directory."""
    return _FILE_HANDLER.named_lock(base_dir, name)


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict using dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        joined = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, joined))
        else:
            flat[joined] = value
    return flat


def unflatten_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted keys into nested dicts."""
    nested: dict[str, Any] = {}
    for key, value in data.items():
        current = nested
        parts = str(key).split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def apply_dotted_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a dotted-key patch onto a nested dict."""
    merged = json.loads(json.dumps(base))
    for key, value in patch.items():
        current = merged
        parts = str(key).split(".")
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return merged


def pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a `{value}` placeholder pattern into a regex."""
    escaped = re.escape(pattern).replace(r"\{value\}", r"(?P<value>[-+]?\d*\.?\d+)")
    return re.compile(escaped)
