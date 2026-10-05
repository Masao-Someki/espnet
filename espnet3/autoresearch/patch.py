"""Dotted-key config patches: apply, and filter by an allow/deny key list.

Ported (with minor simplification) from ``espnet3/autoresearch`` on
``origin/espnet3/atlas`` (``core/serialization.py``): ``apply_dotted_patch``,
``pattern_to_regex``, and ``flatten_dict``.
"""

from __future__ import annotations

import json
import re
from fnmatch import fnmatch
from typing import Any, Dict, Sequence


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict using dotted keys."""
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        joined = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, joined))
        else:
            flat[joined] = value
    return flat


def apply_dotted_patch(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a dotted-key patch onto a nested dict, indexing into lists by position.

    A dotted key part that is a plain non-negative integer (e.g. ``"0"``) is
    treated as a list index whenever the current container is a list, so keys
    like ``dataset.train.0.transform.transforms.1.apply_prob`` can reach into
    a ``dataset.train: [...]``-style list without clobbering it (and its
    sibling entries) into a dict.
    """
    merged = json.loads(json.dumps(base))

    def _set_child(container, part: str, ensure_kind: type):
        if isinstance(container, list):
            idx = int(part)
            while len(container) <= idx:
                container.append({} if ensure_kind is dict else [])
            if not isinstance(container[idx], (dict, list)):
                container[idx] = {} if ensure_kind is dict else []
            return container[idx]
        if part not in container or not isinstance(container[part], (dict, list)):
            container[part] = {} if ensure_kind is dict else []
        return container[part]

    for key, value in patch.items():
        current = merged
        parts = str(key).split(".")
        for part in parts[:-1]:
            current = _set_child(current, part, dict)
        last = parts[-1]
        if isinstance(current, list):
            idx = int(last)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            current[last] = value
    return merged


def pattern_to_regex(pattern: str) -> "re.Pattern[str]":
    """Convert a `{value}` placeholder pattern into a regex."""
    escaped = re.escape(pattern).replace(r"\{value\}", r"(?P<value>[-+]?\d*\.?\d+)")
    return re.compile(escaped)


def sanitize_patch(
    patch: Dict[str, Any],
    allowed_keys: Sequence[str],
    denied_keys: Sequence[str],
) -> Dict[str, Any]:
    """Filter a flat, dotted-key patch to only allowed, non-denied keys.

    Each dotted key in ``patch`` is matched against ``denied_keys`` first
    (``fnmatch``-style globs, e.g. ``"exp_dir"`` or ``"dataset.*"``): a match
    there drops the key regardless of ``allowed_keys``. Otherwise the key is
    kept only if it matches at least one entry in ``allowed_keys``; a key
    that matches neither list is dropped (default-deny).

    Args:
        patch: Flat dict of dotted key -> new value (e.g. an agent's proposed
            ``config_patch``).
        allowed_keys: Glob patterns a key must match to be kept.
        denied_keys: Glob patterns that always drop a key, even if it also
            matches ``allowed_keys``.

    Returns:
        Dict[str, Any]: The subset of ``patch`` whose keys passed the filter.
        Empty if nothing passed (callers should treat this as a failed
        proposal, per the design's "empty patch -> failed trial" rule).
    """
    sanitized: Dict[str, Any] = {}
    for key, value in patch.items():
        if any(fnmatch(key, deny) for deny in denied_keys):
            continue
        if any(fnmatch(key, allow) for allow in allowed_keys):
            sanitized[key] = value
    return sanitized
