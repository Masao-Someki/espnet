"""No-op agent."""

from __future__ import annotations

from espnet3.autoresearch.agents.interface import AgentRequest, AgentResponse


class NullAgentClient:
    """Always returns a failure/no-op response."""

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        return AgentResponse(
            status="failure",
            message="NullAgentClient is configured.",
            content="",
        )
