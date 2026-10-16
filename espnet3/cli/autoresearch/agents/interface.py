"""Agent interface."""

from __future__ import annotations

from typing import Protocol


from espnet3.cli.autoresearch.core.models import AgentRequest, AgentResponse


class AgentClient(Protocol):
    """Agent protocol."""

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        """Run the agent."""
