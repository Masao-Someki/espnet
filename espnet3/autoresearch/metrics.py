"""Read a trial's score from `metric.source` (first readable entry wins).

Ported (with a simplified schema matching this design's `metric.source`
list) from ``espnet3/autoresearch`` on ``origin/espnet3/atlas``
(``observers/metrics.py``): the JSON dotted-key lookup and the
`{value}`-pattern lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from espnet3.autoresearch.patch import pattern_to_regex


class MetricNotFoundError(Exception):
    """Raised when no `metric.source` entry could be read for a trial."""


def _extract_from_json(path: Path, key: str) -> Any:
    data = json.loads(path.read_text(encoding="utf-8"))
    current: Any = data
    if isinstance(current, dict) and key in current:
        return current[key]
    parts = str(key).split(".")
    index = 0
    while index < len(parts):
        if not isinstance(current, dict):
            raise MetricNotFoundError(f"Missing JSON key '{key}' in {path}")
        matched = False
        # Longest-first: a dotted key segment may itself contain dots (e.g.
        # "espnet3.systems.esp2_asr.metrics.wer.WER").
        for end in range(len(parts), index, -1):
            candidate = ".".join(parts[index:end])
            if candidate in current:
                current = current[candidate]
                index = end
                matched = True
                break
        if not matched:
            raise MetricNotFoundError(f"Missing JSON key '{key}' in {path}")
    return current


def _extract_from_pattern(path: Path, pattern: str) -> float:
    regex = pattern_to_regex(pattern)
    text = path.read_text(encoding="utf-8", errors="replace")
    match = regex.search(text)
    if match is None:
        raise MetricNotFoundError(f"Pattern '{pattern}' not found in {path}")
    return float(match.group("value"))


def read_metric(source: Sequence[Mapping[str, Any]]) -> float:
    """Return the first readable score from `source` (already-rendered paths).

    Args:
        source: `metric.source` entries, in priority order, with `path`
            already placeholder-rendered to a concrete file path, and either
            a `key` (JSON dotted key) or a `pattern` (a `{value}` template).

    Returns:
        float: The extracted score.

    Raises:
        MetricNotFoundError: If no entry's file exists and reads successfully.
    """
    errors = []
    for entry in source:
        path = Path(entry["path"])
        if not path.exists():
            errors.append(f"{path}: not found")
            continue
        try:
            if "key" in entry and entry["key"] is not None:
                return float(_extract_from_json(path, entry["key"]))
            if "pattern" in entry and entry["pattern"] is not None:
                return _extract_from_pattern(path, entry["pattern"])
            errors.append(f"{path}: source entry has neither 'key' nor 'pattern'")
        except MetricNotFoundError as exc:
            errors.append(str(exc))
    raise MetricNotFoundError(
        "No metric.source entry could be read: " + "; ".join(errors)
    )
