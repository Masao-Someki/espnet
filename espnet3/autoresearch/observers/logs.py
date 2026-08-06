"""Log summarization helpers."""

from __future__ import annotations

import csv
from pathlib import Path


def read_log_tail(path: Path, max_lines: int = 40) -> str:
    """Return the tail of a text log if it exists."""
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def read_csv_tail(path: Path, max_rows: int = 10) -> str:
    """Return the tail rows of a CSV file, including the header."""
    if not path.is_file():
        return ""
    rows = list(csv.reader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    if not rows:
        return ""
    head = rows[0]
    tail = rows[-max_rows:]
    lines = [",".join(head)]
    for row in tail:
        lines.append(",".join(row))
    return "\n".join(lines)


def summarize_recent_trial_curves(
    study_dir: Path,
    exclude_trial_id: str | None = None,
    max_trials: int = 5,
    max_rows: int = 5,
) -> str:
    """Collect short summaries of recent trial curve CSVs."""
    trials_dir = Path(study_dir) / "trials"
    if not trials_dir.is_dir():
        return ""
    summaries: list[str] = []
    trial_dirs = sorted(
        [path for path in trials_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    for trial_dir in reversed(trial_dirs):
        if exclude_trial_id and trial_dir.name == exclude_trial_id:
            continue
        curve_path = trial_dir / "training_metrics.csv"
        if not curve_path.is_file():
            continue
        tail = read_csv_tail(curve_path, max_rows=max_rows)
        if not tail:
            continue
        summaries.append(
            "\n".join(
                [
                    f"## {trial_dir.name}",
                    f"path: {curve_path}",
                    tail,
                ]
            )
        )
        if len(summaries) >= max_trials:
            break
    return "\n\n".join(summaries)
