"""SQLite-backed study state."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.serialization import append_text, dumps_json, loads_json, utc_now
from espnet3.autoresearch.core.trial import Trial
from espnet3.autoresearch.reports.csv_writer import sync_trials_csv


@dataclass
class Study:
    """Study record."""

    study_id: str
    study_dir: str
    status: str
    created_at: str
    updated_at: str
    current_best_trial_id: str | None = None
    agent_session_id: str | None = None


@dataclass
class JobRecord:
    """Persistent job record."""

    job_id: str
    study_id: str
    trial_id: str | None
    node_run_id: str | None
    backend: str
    external_id: str | None
    status: str
    command_json: str
    resources_json: str | None
    workdir: str | None
    stdout_path: str | None
    stderr_path: str | None
    job_type: str
    created_at: str
    updated_at: str


class StateStore:
    """Persistent AutoResearch state store."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS studies (
                  study_id TEXT PRIMARY KEY,
                  study_dir TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  current_best_trial_id TEXT,
                  agent_session_id TEXT
                );
                CREATE TABLE IF NOT EXISTS trials (
                  trial_id TEXT PRIMARY KEY,
                  study_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  parent_trial_id TEXT,
                  config_patch_json TEXT,
                  resolved_config_path TEXT,
                  rationale TEXT,
                  expected_effect TEXT,
                  risk TEXT,
                  metrics_json TEXT,
                  score REAL,
                  score_name TEXT,
                  decision TEXT,
                  reflection TEXT,
                  next_suggestion TEXT,
                  attempt_count INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS node_runs (
                  run_id TEXT PRIMARY KEY,
                  study_id TEXT NOT NULL,
                  node_name TEXT NOT NULL,
                  trial_id TEXT,
                  status TEXT NOT NULL,
                  attempt INTEGER NOT NULL DEFAULT 0,
                  started_at TEXT,
                  finished_at TEXT,
                  result_status TEXT,
                  result_payload_json TEXT,
                  slurm_job_id TEXT,
                  local_pid INTEGER
                );
                CREATE TABLE IF NOT EXISTS jobs (
                  job_id TEXT PRIMARY KEY,
                  study_id TEXT NOT NULL,
                  trial_id TEXT,
                  node_run_id TEXT,
                  backend TEXT NOT NULL,
                  external_id TEXT,
                  status TEXT NOT NULL,
                  command_json TEXT NOT NULL,
                  resources_json TEXT,
                  workdir TEXT,
                  stdout_path TEXT,
                  stderr_path TEXT,
                  job_type TEXT NOT NULL DEFAULT 'trial',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  study_id TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  payload_json TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "job_type" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN job_type TEXT NOT NULL DEFAULT 'trial'"
                )
            study_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(studies)").fetchall()
            }
            if "agent_session_id" not in study_columns:
                conn.execute("ALTER TABLE studies ADD COLUMN agent_session_id TEXT")
            trial_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(trials)").fetchall()
            }
            if "codex_thread_id" not in trial_columns:
                conn.execute("ALTER TABLE trials ADD COLUMN codex_thread_id TEXT")
            if "extra_stages_json" not in trial_columns:
                conn.execute("ALTER TABLE trials ADD COLUMN extra_stages_json TEXT")

    def create_or_load_study(self, study_id: str, study_dir: Path) -> Study:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM studies WHERE study_id = ?",
                (study_id,),
            ).fetchone()
            now = utc_now()
            if row is None:
                conn.execute(
                    "INSERT INTO studies "
                    "(study_id, study_dir, status, created_at, updated_at, current_best_trial_id, "
                    "agent_session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (study_id, str(study_dir), "initialized", now, now, None, None),
                )
                return Study(study_id, str(study_dir), "initialized", now, now, None, None)
            return Study(**dict(row))

    def get_study(self, study_id: str) -> Study:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM studies WHERE study_id = ?",
                (study_id,),
            ).fetchone()
        return Study(**dict(row))

    def update_study_status(self, study_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE studies SET status = ?, updated_at = ? WHERE study_id = ?",
                (status, utc_now(), study_id),
            )

    def get_agent_session_id(self, study_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT agent_session_id FROM studies WHERE study_id = ?", (study_id,)
            ).fetchone()
        return str(row["agent_session_id"]) if row and row["agent_session_id"] else None

    def set_agent_session_id(self, study_id: str, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE studies SET agent_session_id = ?, updated_at = ? WHERE study_id = ?",
                (session_id, utc_now(), study_id),
            )

    def set_best_trial(self, study_id: str, trial_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE studies SET current_best_trial_id = ?, updated_at = ? "
                "WHERE study_id = ?",
                (trial_id, utc_now(), study_id),
            )

    def _row_to_trial(self, row) -> Trial:
        data = dict(row)
        return Trial(
            trial_id=data["trial_id"],
            study_id=data["study_id"],
            status=data["status"],
            config_patch=loads_json(data["config_patch_json"] or "{}"),
            resolved_config_path=Path(data["resolved_config_path"])
            if data["resolved_config_path"]
            else None,
            rationale=data["rationale"] or "",
            expected_effect=data["expected_effect"] or "",
            risk=data["risk"] or "",
            score=data["score"],
            score_name=data["score_name"],
            decision=data["decision"],
            parent_trial_id=data["parent_trial_id"],
            attempt_count=int(data["attempt_count"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metrics=loads_json(data["metrics_json"] or "{}"),
            reflection=data["reflection"] or "",
            next_suggestion=data["next_suggestion"] or "",
            codex_thread_id=data.get("codex_thread_id"),
            extra_stages=loads_json(data.get("extra_stages_json") or "[]"),
        )

    def create_trial(self, trial: Trial) -> Trial:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trials (
                  trial_id, study_id, status, parent_trial_id, config_patch_json,
                  resolved_config_path, rationale, expected_effect, risk,
                  metrics_json, score, score_name, decision, reflection,
                  next_suggestion, attempt_count, created_at, updated_at,
                  codex_thread_id, extra_stages_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.trial_id,
                    trial.study_id,
                    trial.status,
                    trial.parent_trial_id,
                    dumps_json(trial.config_patch, indent=None),
                    str(trial.resolved_config_path) if trial.resolved_config_path else None,
                    trial.rationale,
                    trial.expected_effect,
                    trial.risk,
                    dumps_json(trial.metrics, indent=None),
                    trial.score,
                    trial.score_name,
                    trial.decision,
                    trial.reflection,
                    trial.next_suggestion,
                    trial.attempt_count,
                    trial.created_at,
                    trial.updated_at,
                    trial.codex_thread_id,
                    dumps_json(trial.extra_stages, indent=None),
                ),
            )
        self._sync_trials_csv(Path(self.get_study(trial.study_id).study_dir), trial.study_id)
        return trial

    def update_trial(self, trial: Trial) -> None:
        trial.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trials SET
                  status = ?, parent_trial_id = ?, config_patch_json = ?,
                  resolved_config_path = ?, rationale = ?, expected_effect = ?,
                  risk = ?, metrics_json = ?, score = ?, score_name = ?,
                  decision = ?, reflection = ?, next_suggestion = ?,
                  attempt_count = ?, updated_at = ?, codex_thread_id = ?,
                  extra_stages_json = ?
                WHERE trial_id = ?
                """,
                (
                    trial.status,
                    trial.parent_trial_id,
                    dumps_json(trial.config_patch, indent=None),
                    str(trial.resolved_config_path) if trial.resolved_config_path else None,
                    trial.rationale,
                    trial.expected_effect,
                    trial.risk,
                    dumps_json(trial.metrics, indent=None),
                    trial.score,
                    trial.score_name,
                    trial.decision,
                    trial.reflection,
                    trial.next_suggestion,
                    trial.attempt_count,
                    trial.updated_at,
                    trial.codex_thread_id,
                    dumps_json(trial.extra_stages, indent=None),
                    trial.trial_id,
                ),
            )
        self._sync_trials_csv(Path(self.get_study(trial.study_id).study_dir), trial.study_id)

    def get_trial(self, trial_id: str) -> Trial:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trials WHERE trial_id = ?",
                (trial_id,),
            ).fetchone()
        return self._row_to_trial(row)

    def list_trials(self, study_id: str, status: str | None = None) -> list[Trial]:
        query = "SELECT * FROM trials WHERE study_id = ?"
        params: list[Any] = [study_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at, trial_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_trial(row) for row in rows]

    def get_best_trial(self, study_id: str, metric: str, mode: str) -> Trial | None:
        trials = [
            trial
            for trial in self.list_trials(study_id)
            if trial.score is not None and trial.score_name == metric
        ]
        if not trials:
            return None
        reverse = mode == "max"
        trials.sort(key=lambda trial: float(trial.score), reverse=reverse)
        return trials[0]

    def create_node_run(
        self,
        run_id: str,
        study_id: str,
        node_name: str,
        trial_id: str | None = None,
        attempt: int = 0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO node_runs (
                  run_id, study_id, node_name, trial_id, status, attempt,
                  started_at, finished_at, result_status, result_payload_json,
                  slurm_job_id, local_pid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    study_id,
                    node_name,
                    trial_id,
                    "pending",
                    attempt,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )

    def begin_node_run(self, run_id: str, pid: int | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE node_runs SET status = ?, started_at = ?, local_pid = ? "
                "WHERE run_id = ?",
                ("running", utc_now(), pid, run_id),
            )

    def finish_node_run(self, run_id: str, result: StageResult) -> None:
        status = "completed"
        if result.status in {"failure", "timeout", "retry"}:
            status = "failed"
        with self._connect() as conn:
            conn.execute(
                "UPDATE node_runs SET status = ?, finished_at = ?, result_status = ?, "
                "result_payload_json = ? WHERE run_id = ?",
                (status, utc_now(), result.status, result.to_json(), run_id),
            )

    def get_pending_node_runs(self, study_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM node_runs WHERE study_id = ? AND status = 'pending' "
                "ORDER BY started_at IS NOT NULL, run_id",
                (study_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_node_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM node_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_node_runs(self, study_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM node_runs WHERE study_id = ? ORDER BY run_id",
                (study_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_job(self, job: JobRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.study_id,
                    job.trial_id,
                    job.node_run_id,
                    job.backend,
                    job.external_id,
                    job.status,
                    job.command_json,
                    job.resources_json,
                    job.workdir,
                    job.stdout_path,
                    job.stderr_path,
                    job.job_type,
                    job.created_at,
                    job.updated_at,
                ),
            )

    def update_job_status(
        self,
        job_id: str,
        status: str,
        external_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, external_id = COALESCE(?, external_id), "
                "updated_at = ? WHERE job_id = ?",
                (status, external_id, utc_now(), job_id),
            )

    def list_active_jobs(self, study_id: str) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE study_id = ? AND status IN "
                "('submitted', 'pending', 'running', 'unknown') ORDER BY created_at",
                (study_id,),
            ).fetchall()
        return [JobRecord(**dict(row)) for row in rows]

    def create_tick_job(self, study_id: str, handle, stdout_path: str, stderr_path: str) -> None:
        """Record a scheduled tick job."""
        metadata = handle.metadata or {}
        self.create_job(
            JobRecord(
                job_id=str(metadata["job_id"]),
                study_id=study_id,
                trial_id=metadata.get("trial_id"),
                node_run_id=None,
                backend=handle.backend,
                external_id=handle.external_id,
                status="submitted",
                command_json=dumps_json(
                    {
                        "command": metadata.get("command", []),
                        "metadata": metadata,
                    },
                    indent=None,
                ),
                resources_json=dumps_json(metadata.get("resources", {}), indent=None),
                workdir=str(metadata.get("workdir")) if metadata.get("workdir") else None,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                job_type=str(metadata.get("job_type", "tick")),
                created_at=utc_now(),
                updated_at=utc_now(),
            )
        )

    def list_active_tick_jobs(self, study_id: str) -> list[JobRecord]:
        """Return non-terminal tick jobs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE study_id = ? AND job_type = 'tick' "
                "AND status IN ('submitted', 'pending', 'running', 'unknown') "
                "ORDER BY created_at",
                (study_id,),
            ).fetchall()
        return [JobRecord(**dict(row)) for row in rows]

    def compute_trial_gpu_hours(self, study_id: str) -> float:
        """Return total GPU-hours used by trial node runs (completed + in-flight)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT started_at, finished_at FROM node_runs "
                "WHERE study_id = ? AND trial_id IS NOT NULL AND started_at IS NOT NULL",
                (study_id,),
            ).fetchall()
        now = datetime.now(timezone.utc)
        total = 0.0
        for row in rows:
            start = datetime.fromisoformat(row["started_at"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            end_str = row["finished_at"]
            if end_str:
                end = datetime.fromisoformat(end_str)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
            else:
                end = now
            total += max(0.0, (end - start).total_seconds() / 3600.0)
        return total

    def finish_tick_job(self, job_id: str) -> None:
        """Mark a tick job as completed."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                ("completed", utc_now(), job_id),
            )

    def find_job_by_node_run(self, node_run_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE node_run_id = ? ORDER BY created_at DESC LIMIT 1",
                (node_run_id,),
            ).fetchone()
        return JobRecord(**dict(row)) if row else None

    def append_event(self, study_id: str, event_type: str, payload: dict) -> None:
        timestamp = utc_now()
        study_dir = Path(self.get_study(study_id).study_dir)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (study_id, timestamp, event_type, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (study_id, timestamp, event_type, dumps_json(payload, indent=None)),
            )
        append_text(
            study_dir / "events.jsonl",
            dumps_json(
                {
                    "study_id": study_id,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "payload": payload,
                },
                indent=None,
            )
            + "\n",
        )

    def _sync_trials_csv(self, study_dir: Path, study_id: str) -> None:
        sync_trials_csv(self, study_dir, study_id)
