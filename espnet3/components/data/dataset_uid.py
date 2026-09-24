"""Dataset-hash UID generation, on-disk table, and cross-run validation.

ESPnet3's stable identifier for a dataset item is ``"<8-hex-char dataset
hash>:<integer position within that dataset>"`` (see ``format_uid``). The
hash identifies *which dataset configuration* produced the item, not the
item itself, so generating and resolving UIDs never requires a structure
sized to the number of items in a dataset -- only one entry per dataset.

Typical flow:

1. ``DataOrganizer`` computes ``compute_entry_hash`` for every ``dataset:``
   entry of a split and calls ``check_entry_hashes`` to catch duplicates or
   32-bit hash collisions before building the split's ``CombinedDataset``.
2. ``collect_stats`` writes one ``DatasetUidEntry`` per dataset (via
   ``write_uid_table``) into ``<stats_dir>/<mode>/dataset_uids.json``,
   alongside that split's shape files.
3. At training time, before shape-file batches are used,
   ``validate_against_uid_table`` reads that table back (via
   ``load_uid_table``) and confirms it still matches the current dataset
   configuration, raising a ``RuntimeError`` that tells the user which
   dataset changed and to re-run ``collect_stats`` if it does not.

Only keys ending in ``_dir`` (``recipe_dir``, ``data_dir``,
``cache.cache_dir``, ...) are excluded from the hash, including inside
nested mappings -- pointing the same dataset configuration at a different
filesystem location does not change its identity.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

UID_HASH_LENGTH = 8
EXCLUDED_KEY_SUFFIX = "_dir"
UID_TABLE_FILENAME = "dataset_uids.json"
UID_TABLE_FORMAT = 1
_UID_PATTERN = re.compile(r"^([0-9a-f]{8}):([0-9]+)$")
_MAX_DIFF_ITEMS = 5


def strip_dir_keys(obj: Any) -> Any:
    """Return a deep copy of ``obj`` with every ``*_dir`` mapping key removed.

    Applies recursively to nested mappings and sequences, so a key such as
    ``data_src_args.cache.cache_dir`` is excluded just like a top-level
    ``recipe_dir``. Used to make the dataset-hash UID insensitive to where a
    dataset's files live on disk.

    Args:
        obj (Any): A dataset entry (or any nested value inside one) as
            produced by ``OmegaConf.to_container`` -- plain ``dict``/
            ``list``/scalars, no live objects.

    Returns:
        Any: ``obj`` with keys ending in ``_dir`` removed from every mapping
        at any nesting depth. Non-mapping, non-sequence values are returned
        unchanged (not copied).
    """
    if isinstance(obj, Mapping):
        return {
            key: strip_dir_keys(value)
            for key, value in obj.items()
            if not (isinstance(key, str) and key.endswith(EXCLUDED_KEY_SUFFIX))
        }
    if isinstance(obj, (list, tuple)):
        return [strip_dir_keys(value) for value in obj]
    return obj


def _json_default(value: Any) -> Any:
    """Support the surrounding workflow.

    Raises:
        TypeError: For any value that is not a ``pathlib.Path`` or
            ``enum.Enum``. Deliberately does not fall back to ``repr()``:
            an object's ``repr`` usually embeds its memory address, which
            would make the hash different on every run.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    raise TypeError(
        f"dataset entry field of type {type(value).__name__} cannot be "
        "hashed; express it in config (e.g. a _target_ dict) instead of a "
        "live object."
    )


def canonicalize_entry(entry: Mapping[str, Any]) -> str:
    """Serialize a dataset entry to the canonical JSON the hash is computed from.

    Args:
        entry (Mapping[str, Any]): One ``dataset:`` entry as received by
            ``DataOrganizer`` (``data_src``, ``data_src_args``, ``transform``,
            ``name``, ...; already a plain ``dict``, not a Hydra object).

    Returns:
        str: ``entry`` with ``*_dir`` keys stripped (``strip_dir_keys``),
        serialized as JSON with sorted keys and no extra whitespace, so that
        two entries differing only in key order or in a ``*_dir`` value
        canonicalize to the same string.

    Raises:
        TypeError: If ``entry`` contains a value that is not JSON-serializable
            and not a ``Path``/``Enum`` (see ``_json_default``).
    """
    return json.dumps(
        strip_dir_keys(entry),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def compute_entry_hash(entry: Mapping[str, Any]) -> str:
    """Return the 8-hex-char UID prefix for a dataset entry.

    Args:
        entry (Mapping[str, Any]): One ``dataset:`` entry, as passed to
            ``canonicalize_entry``.

    Returns:
        str: The first ``UID_HASH_LENGTH`` hex characters of the SHA-256
        digest of ``canonicalize_entry(entry)``. Two entries with the same
        canonical JSON (i.e. equal ignoring ``*_dir`` keys) always produce
        the same prefix; unrelated entries collide only with 32-bit
        probability (see ``check_entry_hashes``).
    """
    digest = hashlib.sha256(canonicalize_entry(entry).encode("utf-8")).hexdigest()
    return digest[:UID_HASH_LENGTH]


def format_uid(prefix: str, position: int) -> str:
    """Return the UID string for a dataset hash prefix and item position."""
    return f"{prefix}:{position}"


def parse_uid(uid: str) -> Optional[Tuple[str, int]]:
    """Parse a UID string produced by ``format_uid``.

    Args:
        uid (str): A candidate UID string.

    Returns:
        Optional[Tuple[str, int]]: ``(prefix, position)`` if ``uid`` matches
        ``"<8 lowercase hex chars>:<non-negative integer>"``, else ``None``
        (e.g. for a shard-local label such as ``"a1b2c3d4@s1:5"``, a plain
        integer-index string, or an unrelated string-mode key).
    """
    match = _UID_PATTERN.fullmatch(uid)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


@dataclass(frozen=True)
class DatasetUidEntry:
    """One dataset's row in a ``dataset_uids.json`` table.

    Attributes:
        uid_prefix (str): The 8-hex-char hash from ``compute_entry_hash``.
        label (str): Human-readable identifier for error messages --
            ``entry["name"]``, else ``entry["data_src"]``, else
            ``f"entry #{i}"``.
        num_items (int): ``len(dataset)`` at the time the table was written
            (or, for the "current" side, right now).
        config (Dict[str, Any]): The entry with ``*_dir`` keys removed
            (``strip_dir_keys(entry)``), stored so a mismatch can be
            explained with a field-level diff.
    """

    uid_prefix: str
    label: str
    num_items: int
    config: Dict[str, Any]


def write_uid_table(
    split_dir: Union[str, Path], entries: Sequence[DatasetUidEntry]
) -> Path:
    """Write a split's dataset-hash UID table next to its shape files.

    Writes atomically (temp file in the same directory, then
    ``os.replace``), so a reader never observes a partially written file.

    Args:
        split_dir (Union[str, Path]): The split directory to write into
            (e.g. ``<stats_dir>/train``), matching where shape files for
            that split live.
        entries (Sequence[DatasetUidEntry]): One entry per dataset in the
            split's ``CombinedDataset``, in that dataset's order.

    Returns:
        Path: The path the table was written to
        (``split_dir / UID_TABLE_FILENAME``).
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": UID_TABLE_FORMAT,
        "datasets": [
            {
                "uid_prefix": entry.uid_prefix,
                "label": entry.label,
                "num_items": entry.num_items,
                "config": entry.config,
            }
            for entry in entries
        ],
    }
    target = split_dir / UID_TABLE_FILENAME
    tmp_path = split_dir / f".{UID_TABLE_FILENAME}.tmp.{os.getpid()}"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, target)
    return target


def load_uid_table(
    split_dir: Union[str, Path],
) -> Optional[List[DatasetUidEntry]]:
    """Read a split's dataset-hash UID table written by ``write_uid_table``.

    Args:
        split_dir (Union[str, Path]): The split directory to read from.

    Returns:
        Optional[List[DatasetUidEntry]]: The table's entries, in their
        original order, or ``None`` if ``split_dir / UID_TABLE_FILENAME``
        does not exist (e.g. shape files from before dataset-hash UIDs, or
        a ``CombinedDataset`` built without ``uid_entries``).

    Raises:
        RuntimeError: If the file exists but cannot be parsed as JSON, has
            an unsupported ``"format"``, or is missing expected fields.
    """
    split_dir = Path(split_dir)
    target = split_dir / UID_TABLE_FILENAME
    if not target.is_file():
        return None

    try:
        with target.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Failed to read {target}: {e}") from e

    if not isinstance(payload, dict) or payload.get("format") != UID_TABLE_FORMAT:
        got = (
            payload.get("format")
            if isinstance(payload, dict)
            else type(payload).__name__
        )
        raise RuntimeError(
            f"{target} has an unsupported format (expected "
            f"format={UID_TABLE_FORMAT}, got {got!r}); re-run collect_stats "
            "to regenerate it."
        )

    try:
        return [
            DatasetUidEntry(
                uid_prefix=row["uid_prefix"],
                label=row["label"],
                num_items=row["num_items"],
                config=row["config"],
            )
            for row in payload["datasets"]
        ]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"{target} is malformed: {e}") from e


def check_entry_hashes(entries: Sequence[Tuple[str, str, str]]) -> None:
    """Raise if any two dataset entries in a split collide or duplicate.

    Args:
        entries (Sequence[Tuple[str, str, str]]): ``(uid_prefix,
            canonical_json, label)`` for every entry in one split, in entry
            order. ``canonical_json`` is ``canonicalize_entry(entry)``.

    Raises:
        ValueError: If two entries produce the same ``uid_prefix``. The
            message distinguishes a true duplicate (identical
            ``canonical_json``, i.e. the same data listed twice -- possibly
            differing only in ``*_dir`` keys) from a genuine 32-bit hash
            collision between two different configurations.
    """
    seen: Dict[str, Tuple[str, str]] = {}
    for prefix, canonical, label in entries:
        if prefix in seen:
            prev_canonical, prev_label = seen[prefix]
            if prev_canonical == canonical:
                raise ValueError(
                    f'dataset entries "{prev_label}" and "{label}" have '
                    "identical configuration (ignoring *_dir keys); the "
                    "same data would be listed twice. Remove one of them."
                )
            raise ValueError(
                f'dataset entries "{prev_label}" and "{label}" differ but '
                f"hash to the same UID prefix {prefix} (32-bit collision). "
                'Rename one entry (e.g. change its "name") to separate '
                "them."
            )
        seen[prefix] = (canonical, label)


def _diff_config_paths(
    old: Any, new: Any, max_items: int = _MAX_DIFF_ITEMS
) -> List[str]:
    """Support the surrounding workflow.

    Recursively compares two (already ``strip_dir_keys``-filtered) config
    values and returns up to ``max_items`` ``"dotted.path: old -> new"``
    strings describing the differences, walked in sorted-key order for
    determinism.
    """
    changes: List[str] = []

    def walk(old_value: Any, new_value: Any, path: str) -> None:
        if len(changes) >= max_items:
            return
        if isinstance(old_value, Mapping) and isinstance(new_value, Mapping):
            for key in sorted(set(old_value) | set(new_value)):
                if len(changes) >= max_items:
                    return
                sub_path = f"{path}.{key}" if path else str(key)
                if key not in old_value:
                    changes.append(f"{sub_path}: (missing) -> {new_value[key]!r}")
                elif key not in new_value:
                    changes.append(f"{sub_path}: {old_value[key]!r} -> (missing)")
                else:
                    walk(old_value[key], new_value[key], sub_path)
        elif old_value != new_value:
            changes.append(f"{path}: {old_value!r} -> {new_value!r}")

    walk(old, new, "")
    return changes[:max_items]


def _find_matching_stale_entry(
    missing: DatasetUidEntry, stale: Sequence[DatasetUidEntry]
) -> Optional[DatasetUidEntry]:
    """Support the surrounding workflow."""
    for candidate in stale:
        if candidate.label == missing.label:
            return candidate
    data_src = missing.config.get("data_src")
    if data_src is not None:
        for candidate in stale:
            if candidate.config.get("data_src") == data_src:
                return candidate
    return None


def _build_changed_message(
    split_dir: Union[str, Path],
    missing: Sequence[DatasetUidEntry],
    stale: Sequence[DatasetUidEntry],
) -> str:
    """Support the surrounding workflow."""
    lines = [
        "The dataset configuration no longer matches the one collect_stats "
        f"used for {split_dir}."
    ]
    if missing:
        lines.append("  changed / new in the current config:")
        for entry in missing:
            match = _find_matching_stale_entry(entry, stale)
            if match is not None:
                diffs = _diff_config_paths(match.config, entry.config)
                detail = "; ".join(diffs) if diffs else "configuration changed"
            else:
                detail = "new dataset entry"
            lines.append(f'    - "{entry.label}" (hash {entry.uid_prefix}): {detail}')
    if stale:
        lines.append("  present in stats_dir but not in the current config:")
        for entry in stale:
            lines.append(f'    - "{entry.label}" (hash {entry.uid_prefix})')
    lines.append(
        "Re-run the collect_stats stage (or point stats_dir at the "
        "directory produced for this configuration)."
    )
    return "\n".join(lines)


def validate_against_uid_table(
    split_dir: Union[str, Path],
    current: Optional[Sequence[DatasetUidEntry]],
) -> None:
    """Confirm a split's on-disk UID table still matches the current config.

    Called before shape-file batches (produced by an earlier
    ``collect_stats`` run) are trusted, so a stale ``stats_dir`` fails fast
    with an actionable error instead of a ``KeyError`` mid-training.

    Args:
        split_dir (Union[str, Path]): The split directory whose shape files
            are about to be used (e.g. the directory a ``shape_files`` entry
            lives in).
        current (Optional[Sequence[DatasetUidEntry]]): The current
            ``CombinedDataset``'s ``uid_entries``. ``None`` means the
            dataset was built without dataset-hash UIDs (legacy/direct
            construction); validation is then skipped entirely, matching
            ``CombinedDataset``'s own ``str(idx)`` fallback.

    Raises:
        RuntimeError: If ``current`` is not ``None`` and any of: no table
            exists at ``split_dir`` (``dataset_uids.json`` missing or
            unreadable -- see ``load_uid_table``); the table's set of
            dataset hashes differs from ``current``'s (a dataset's
            configuration changed, was added, or was removed); or a
            dataset present in both has a different ``num_items`` (its
            contents changed without a configuration change).
    """
    if current is None:
        return

    table = load_uid_table(split_dir)
    if table is None:
        raise RuntimeError(
            f"{Path(split_dir) / UID_TABLE_FILENAME} not found. These "
            "shape files were written by an older collect_stats (before "
            "dataset-hash UIDs) or by a different tool. Re-run the "
            "collect_stats stage for this stats_dir."
        )

    current_by_prefix = {entry.uid_prefix: entry for entry in current}
    table_by_prefix = {entry.uid_prefix: entry for entry in table}
    missing = [e for e in current if e.uid_prefix not in table_by_prefix]
    stale = [e for e in table if e.uid_prefix not in current_by_prefix]
    if missing or stale:
        raise RuntimeError(_build_changed_message(split_dir, missing, stale))

    for prefix, current_entry in current_by_prefix.items():
        table_entry = table_by_prefix[prefix]
        if current_entry.num_items != table_entry.num_items:
            raise RuntimeError(
                f'dataset "{current_entry.label}" (hash {prefix}) has '
                f"{current_entry.num_items} items now but "
                f"{table_entry.num_items} when collect_stats ran; its "
                "contents changed. Re-run collect_stats."
            )
