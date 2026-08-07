"""AutoResearch stage base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.serialization import atomic_write_text, named_lock, utc_now


class AutoResearchStage(ABC):
    """Base class implementing idempotent stage execution."""

    study_lock_name: str | None = None

    def execute(self, context) -> StageResult:
        marker_dir = context.artifact_dir()
        marker_dir.mkdir(parents=True, exist_ok=True)
        success_path = marker_dir / ".stage_success.json"
        failure_path = marker_dir / ".stage_failure.json"
        started_path = marker_dir / ".stage_started.json"
        if success_path.exists() and not bool(context.config.get("force", False)):
            return StageResult.from_json(success_path.read_text(encoding="utf-8"))
        atomic_write_text(started_path, f'{{"started_at": "{utc_now()}"}}')
        if self.study_lock_name is None:
            result = self.run(context)
        else:
            with named_lock(context.study_dir, self.study_lock_name):
                result = self.run(context)
        target = success_path
        if result.status in {"failure", "timeout", "retry"}:
            target = failure_path
        atomic_write_text(target, result.to_json())
        return result

    @abstractmethod
    def run(self, context) -> StageResult:
        """Run the stage."""
