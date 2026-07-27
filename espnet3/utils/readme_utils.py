"""Shared helpers for building README markdown and environment metadata.

Centralizes the pieces that both the ``measure`` stage (a lightweight
Environments + Results README written next to ``metrics.json``) and the
``pack_model`` stage (a richer README bundled with a packed model) need, so
version/git metadata and the results table are not gathered twice.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

import espnet2
from espnet3.utils.logging_utils import get_git_metadata


def get_environment_info(cwd: Path | None = None) -> dict[str, str]:
    """Gather version and git metadata shared across READMEs and meta.yaml.

    Args:
        cwd: Directory to resolve git metadata from. Defaults to the current
            working directory.

    Returns:
        Dict with ``date``, ``python``, ``torch``, ``espnet`` version strings
        and git metadata (``git_commit``, ``git_short_commit``,
        ``git_branch``, ``git_dirty``, ``git_origin``). Git fields are ``""``
        when the value could not be determined (e.g. not a git repository).

    Examples:
        >>> info = get_environment_info()
        >>> info["torch"]
        '2.1.2+cu118'
    """
    git_meta = get_git_metadata(cwd)
    return {
        "date": datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
        "python": sys.version.replace("\n", " "),
        "torch": str(torch.__version__),
        "espnet": str(espnet2.__version__),
        "git_commit": git_meta.get("commit") or "",
        "git_short_commit": git_meta.get("short_commit") or "",
        "git_branch": git_meta.get("branch") or "",
        "git_dirty": git_meta.get("worktree") or "",
        "git_origin": git_meta.get("origin_url") or "",
    }


def build_environments_section(env_info: dict[str, str]) -> str:
    """Render a markdown ``## Environments`` block.

    Args:
        env_info: Dict returned by :func:`get_environment_info`.

    Returns:
        Markdown text for the Environments section.

    Examples:
        >>> section = build_environments_section(get_environment_info())
        >>> section.splitlines()[0]
        '## Environments'
    """
    return "\n".join(
        [
            "## Environments",
            "",
            f"- date: `{env_info['date']}`",
            f"- python version: `{env_info['python']}`",
            f"- espnet version: `espnet {env_info['espnet']}`",
            f"- pytorch version: `pytorch {env_info['torch']}`",
            f"- Git hash: `{env_info['git_commit']}`",
            "",
        ]
    )


def build_results_table(results: dict) -> str:
    """Render a markdown results table from a ``measure()`` results dict.

    Args:
        results: Nested dict keyed by metric class path, then by test set
            name, as returned by ``espnet3.systems.base.metric.measure``.

    Returns:
        Markdown text with a ``## Results`` heading and a table with test
        set names as rows and metric names as columns. Empty string if
        ``results`` has no usable rows or metric keys.

    Examples:
        >>> table = build_results_table(
        ...     {"espnet3.systems.asr.metrics.wer.WER": {"test": {"WER": 5.0}}}
        ... )
        >>> table.splitlines()[0]
        '## Results'
    """
    # rows[test_name][metric_key] = value
    rows: dict[str, dict[str, str]] = {}
    metric_keys: set[str] = set()
    for metric_name, per_test in results.items():
        if not isinstance(per_test, dict):
            continue
        short_name = str(metric_name).rsplit(".", maxsplit=1)[-1]
        for test_name, value in per_test.items():
            rows.setdefault(str(test_name), {})
            if isinstance(value, dict):
                for key, val in value.items():
                    metric_keys.add(key)
                    rows[str(test_name)][key] = str(val)
            else:
                metric_keys.add(short_name)
                rows[str(test_name)][short_name] = str(value)
    if not rows or not metric_keys:
        return ""
    cols = sorted(metric_keys)
    lines = [
        "## Results",
        "",
        "| dataset | " + " | ".join(cols) + " |",
        "| --- | " + " | ".join("---" for _ in cols) + " |",
    ]
    for test in sorted(rows):
        vals = [rows[test].get(col, "") for col in cols]
        lines.append("| " + " | ".join([test] + vals) + " |")
    lines.append("")
    return "\n".join(lines)


def build_results_table_from_file(results_path: Path | None) -> str:
    """Render a markdown results table from a ``metrics.json`` file.

    Args:
        results_path: Path to a ``metrics.json`` file, or ``None``.

    Returns:
        Markdown text from :func:`build_results_table`, or ``""`` if
        ``results_path`` is ``None``, missing, or not valid JSON.

    Examples:
        >>> build_results_table_from_file(Path("exp/run/inference/metrics.json"))
    """
    if results_path is None or not results_path.exists():
        return ""
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return build_results_table(results)


def write_measure_readme(inference_dir: Path, results: dict) -> Path:
    """Write a minimal Environments + Results README.md for the measure stage.

    Overwrites any existing ``README.md`` at the same path. Unlike the
    ``pack_model`` README, this intentionally omits model summary and usage
    sections since the measure stage only has access to scoring results.

    Args:
        inference_dir: Directory containing ``metrics.json``; ``README.md``
            is written alongside it.
        results: Nested dict returned by ``measure()``.

    Returns:
        Path to the written ``README.md``.

    Examples:
        >>> results = measure(metrics_config)
        >>> write_measure_readme(Path(metrics_config.inference_dir), results)
        PosixPath('exp/run/inference/README.md')
    """
    env_info = get_environment_info()
    sections = ["# RESULTS", "", build_environments_section(env_info)]
    results_section = build_results_table(results)
    if results_section:
        sections.append(results_section)
    content = "\n".join(sections).rstrip() + "\n"
    out_path = Path(inference_dir) / "README.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path
