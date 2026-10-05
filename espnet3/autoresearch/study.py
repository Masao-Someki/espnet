"""Study-directory bookkeeping: trial.json, trials.csv, best.json, leaderboard.csv.

Ported (simplified; writes one JSON record per trial instead of an sqlite
database) from `espnet3/autoresearch` on `origin/espnet3/atlas`
(`core/state.py`, `reports/csv_writer.py`): the trial-record idea and the
trials.csv export.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_TRIALS_DIRNAME = "trials"
_TRIAL_RECORD_FILENAME = "trial.json"
_TRIALS_CSV_FIELDS = [
    "trial_id",
    "status",
    "score",
    "patch",
    "rationale",
    "created_at",
    "finished_at",
    "reason",
]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TrialRecord:
    """State of record for one trial (the contents of `trial.json`)."""

    trial_id: str
    status: str  # running | accepted | rejected | failed | skipped
    patch: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    rationale: str = ""
    created_at: str = ""
    finished_at: Optional[str] = None
    reason: str = ""
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain, JSON-serializable dict of this record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialRecord":
        """Reconstruct a `TrialRecord` from a plain dict (e.g. loaded JSON)."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def trials_dir(study_dir: Path) -> Path:
    """Return `<study_dir>/trials`."""
    return study_dir / _TRIALS_DIRNAME


def trial_dir(study_dir: Path, trial_id: str) -> Path:
    """Return `<study_dir>/trials/<trial_id>`."""
    return trials_dir(study_dir) / trial_id


def next_trial_id(study_dir: Path) -> str:
    """Return the next sequential `trial_NNNNNN` id, based on `trials/` contents."""
    existing = trials_dir(study_dir)
    numbers = []
    if existing.is_dir():
        for child in existing.iterdir():
            if child.is_dir() and child.name.startswith("trial_"):
                suffix = child.name[len("trial_") :]
                if suffix.isdigit():
                    numbers.append(int(suffix))
    next_number = max(numbers, default=0) + 1
    return f"trial_{next_number:06d}"


def init_study(study_dir: Path, *, config_yaml_text: str, objective_text: str) -> None:
    """Create `study_dir` and its bookkeeping files (idempotent).

    Args:
        study_dir: The study's root directory (created, with `trials/`,
            if missing).
        config_yaml_text: The resolved `autoresearch.yaml` text to copy in
            as `autoresearch.yaml` (a record of what this study ran with).
        objective_text: Text to write as `program.md` if it does not already
            exist (an existing `program.md` -- e.g. edited by a human -- is
            left untouched).
    """
    study_dir.mkdir(parents=True, exist_ok=True)
    trials_dir(study_dir).mkdir(parents=True, exist_ok=True)
    (study_dir / "autoresearch.yaml").write_text(config_yaml_text, encoding="utf-8")
    program_md = study_dir / "program.md"
    if not program_md.exists():
        program_md.write_text(objective_text, encoding="utf-8")
    trials_csv = study_dir / "trials.csv"
    if not trials_csv.exists():
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_TRIALS_CSV_FIELDS).writeheader()


def write_trial_record(study_dir: Path, record: TrialRecord) -> Path:
    """Write `record` as `<study_dir>/trials/<trial_id>/trial.json`."""
    tdir = trial_dir(study_dir, record.trial_id)
    tdir.mkdir(parents=True, exist_ok=True)
    path = tdir / _TRIAL_RECORD_FILENAME
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def read_trial_record(study_dir: Path, trial_id: str) -> TrialRecord:
    """Read `<study_dir>/trials/<trial_id>/trial.json`."""
    path = trial_dir(study_dir, trial_id) / _TRIAL_RECORD_FILENAME
    return TrialRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_all_trial_records(study_dir: Path) -> List[TrialRecord]:
    """Return every trial's record, sorted by `trial_id`."""
    records = []
    tdir = trials_dir(study_dir)
    if not tdir.is_dir():
        return records
    for child in sorted(tdir.iterdir()):
        record_path = child / _TRIAL_RECORD_FILENAME
        if record_path.is_file():
            records.append(TrialRecord.from_dict(json.loads(record_path.read_text())))
    return records


def mark_interrupted_running_trials(study_dir: Path) -> List[str]:
    """Flip any `status == "running"` trial to `failed(interrupted)`, for resume.

    Returns:
        List[str]: The trial ids that were changed.
    """
    changed = []
    for record in load_all_trial_records(study_dir):
        if record.status == "running":
            record.status = "failed"
            record.reason = "interrupted"
            record.finished_at = utc_now()
            write_trial_record(study_dir, record)
            append_trials_csv(study_dir, record)
            changed.append(record.trial_id)
    return changed


def append_trials_csv(study_dir: Path, record: TrialRecord) -> None:
    """Append one row for `record` to `<study_dir>/trials.csv`."""
    path = study_dir / "trials.csv"
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TRIALS_CSV_FIELDS)
        writer.writerow(
            {
                "trial_id": record.trial_id,
                "status": record.status,
                "score": record.score,
                "patch": json.dumps(record.patch, sort_keys=True),
                "rationale": record.rationale,
                "created_at": record.created_at,
                "finished_at": record.finished_at or "",
                "reason": record.reason,
            }
        )


def read_trials_csv_text(study_dir: Path) -> str:
    """Return the full contents of `<study_dir>/trials.csv` (for agent context)."""
    path = study_dir / "trials.csv"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_best(study_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the contents of `<study_dir>/best.json`, or `None` if absent."""
    path = study_dir / "best.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_best(study_dir: Path, record: TrialRecord, mode: str) -> bool:
    """Update `best.json` if `record` (an accepted trial) beats the current best.

    Args:
        study_dir: The study's root directory.
        record: An accepted trial's record (must have `score` set).
        mode: `"min"` or `"max"`: whether a lower or higher score is better.

    Returns:
        bool: Whether `record` became (or remains, on a tie with an earlier
        trial) the new best.
    """
    current = read_best(study_dir)
    is_better = current is None or (
        record.score < current["score"]
        if mode == "min"
        else record.score > current["score"]
    )
    if is_better:
        (study_dir / "best.json").write_text(
            json.dumps(
                {
                    "trial_id": record.trial_id,
                    "score": record.score,
                    "patch": record.patch,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return is_better


def write_leaderboard(study_dir: Path, mode: str) -> None:
    """Rewrite `<study_dir>/leaderboard.csv`: accepted trials, best score first."""
    accepted = [r for r in load_all_trial_records(study_dir) if r.status == "accepted"]
    accepted.sort(key=lambda r: r.score, reverse=(mode == "max"))
    path = study_dir / "leaderboard.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "score", "patch"])
        writer.writeheader()
        for record in accepted:
            writer.writerow(
                {
                    "trial_id": record.trial_id,
                    "score": record.score,
                    "patch": json.dumps(record.patch, sort_keys=True),
                }
            )
