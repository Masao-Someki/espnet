"""Study-scoped session handling for stateful agent backends."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path

from espnet3.autoresearch.agents.interface import AgentRequest, AgentResponse
from espnet3.autoresearch.core.locking import file_lock


class StudySessionAgentClient:
    """Reuse one agent session across all stages in a study."""

    def __init__(
        self, delegate, state, study_id: str, study_dir: Path, configured_session_id=None
    ) -> None:
        self._delegate = delegate
        self._state = state
        self._study_id = study_id
        self._configured_session_id = str(configured_session_id or "").strip() or None
        self._lock_path = Path(study_dir) / ".lock_agent_session"
        self.session_id = self._configured_session_id or state.get_agent_session_id(study_id)

    @contextlib.contextmanager
    def session_lock(self):
        """Serialize access to the shared agent session across CPU jobs."""
        with file_lock(self._lock_path):
            persisted = self._state.get_agent_session_id(self._study_id)
            self.session_id = self._configured_session_id or persisted
            if self._configured_session_id and self._configured_session_id != persisted:
                self._state.set_agent_session_id(self._study_id, self._configured_session_id)
            yield self.session_id

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        with self.session_lock():
            # A study session deliberately overrides trial-local sessions so every
            # proposal, reflection, and debug action shares one conversation.
            if self.session_id:
                request = replace(request, resume_thread_id=self.session_id)
            response = self._delegate.run(request, artifact_dir=artifact_dir)
            if self.session_id is None and response.thread_id:
                self.session_id = response.thread_id
                self._state.set_agent_session_id(self._study_id, self.session_id)
            return response
