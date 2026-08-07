"""Trial dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TrialStatus = Literal[
    "proposing",
    "proposed",
    "running",
    "completed",
    "failed",
    "timeout",
    "needs_resume",
    "accepted",
    "rejected",
    "cancelled",
    "early_stopped",
]


@dataclass
class Trial:
    """Represents one experiment proposal."""

    trial_id: str
    study_id: str
    status: TrialStatus
    config_patch: dict[str, Any]
    resolved_config_path: Path | None
    rationale: str
    expected_effect: str
    risk: str
    score: float | None
    score_name: str | None
    decision: str | None
    parent_trial_id: str | None
    attempt_count: int
    created_at: str
    updated_at: str
    metrics: dict[str, Any] = field(default_factory=dict)
    reflection: str = ""
    next_suggestion: str = ""
    codex_thread_id: str | None = None
    extra_stages: list[str] = field(default_factory=list)
