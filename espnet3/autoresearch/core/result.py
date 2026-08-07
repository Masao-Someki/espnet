"""Stage result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from espnet3.autoresearch.core.serialization import dumps_json, loads_json

StageStatus = Literal[
    "success",
    "failure",
    "timeout",
    "retry",
    "skip",
    "accepted",
    "rejected",
    "continue",
    "done",
    "no_budget",
    "waiting_agent",
    "waiting_job",
    "terminal",
]


@dataclass
class StageResult:
    """Returned by every AutoResearch stage."""

    status: StageStatus
    message: str = ""
    next_node: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "next_node": self.next_node,
            "payload": self.payload,
            "artifacts": self.artifacts,
        }

    def to_json(self) -> str:
        return dumps_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "StageResult":
        return cls(**loads_json(text))
