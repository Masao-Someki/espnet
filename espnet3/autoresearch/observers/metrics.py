"""Metric extraction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from espnet3.autoresearch.core.errors import MetricNotFoundError
from espnet3.autoresearch.core.serialization import pattern_to_regex


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


def extract_metrics(trial_dir: Path, extraction_rules) -> dict[str, Any]:
    """Extract metrics from files relative to a trial directory."""
    metrics: dict[str, Any] = {}
    for rule in extraction_rules or []:
        rel_path = str(getattr(rule, "path", rule.get("path")))
        matches = sorted(trial_dir.glob(rel_path))
        if not matches:
            continue
        path = matches[0]
        key = getattr(rule, "name", None) or rule.get("name")
        if getattr(rule, "key", None) is not None or rule.get("key") is not None:
            json_key = getattr(rule, "key", None) or rule.get("key")
            value = _extract_from_json(path, json_key)
            aggregate = getattr(rule, "aggregate", None) or (
                rule.get("aggregate") if isinstance(rule, dict) else None
            )
            if aggregate == "mean":
                sub_key = getattr(rule, "sub_key", None) or (
                    rule.get("sub_key") if isinstance(rule, dict) else None
                ) or "WER"
                if not isinstance(value, dict):
                    raise MetricNotFoundError(
                        f"aggregate:mean requires a dict value at key '{json_key}'"
                    )
                values = [
                    float(v[sub_key])
                    for v in value.values()
                    if isinstance(v, dict) and sub_key in v
                ]
                if not values:
                    raise MetricNotFoundError(
                        f"No '{sub_key}' sub-values found under '{json_key}'"
                    )
                value = sum(values) / len(values)
            metrics[key or str(json_key)] = value
        elif (
            getattr(rule, "pattern", None) is not None
            or rule.get("pattern") is not None
        ):
            pattern = getattr(rule, "pattern", None) or rule.get("pattern")
            metrics[key or path.stem] = _extract_from_pattern(path, pattern)
    return metrics


def resolve_primary_score(metrics: dict[str, Any], score_name: str) -> float:
    """Return the score metric as float."""
    if score_name not in metrics:
        raise MetricNotFoundError(f"Score metric '{score_name}' was not extracted.")
    return float(metrics[score_name])
