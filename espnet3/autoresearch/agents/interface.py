"""Agent interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class AgentRequest:
    """Generic request sent to an agent."""

    task: str
    objective: str
    knowledge: str
    repo_context: str
    trial_history_csv: str
    latest_metrics: dict[str, Any]
    latest_logs: dict[str, str]
    allowed_actions: list[str]
    output_schema: dict[str, Any]
    inflight_patches: list[dict] = field(default_factory=list)
    resume_thread_id: str | None = None


@dataclass
class AgentResponse:
    """Structured response from an agent."""

    status: Literal["success", "failure"]
    message: str
    content: str
    structured: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    thread_id: str | None = None
    thread_id: str | None = None


class AgentClient(Protocol):
    """Agent protocol."""

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        """Run the agent."""
