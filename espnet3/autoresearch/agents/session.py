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
        self,
        delegate,
        state,
        study_id: str,
        study_dir: Path,
        recipe_dir: Path,
        configured_session_id=None,
    ) -> None:
        self._delegate = delegate
        self._state = state
        self._study_id = study_id
        self._configured_session_id = str(configured_session_id or "").strip() or None
        self._study_dir = Path(study_dir)
        self._recipe_dir = Path(recipe_dir)
        self._lock_path = self._study_dir / ".lock_agent_session"
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
            request = replace(
                request,
                objective=(
                    "Read study context with tools as needed; do not ask for pasted copies.\n"
                    f"- Objective: {self._study_dir / 'program.md'}\n"
                    f"- Knowledge pack: {self._study_dir / 'knowledge' / 'knowledge_pack.md'}\n"
                    f"- Trial history: {self._study_dir / 'trials.csv'}\n"
                    f"- Resolved AutoResearch config: {self._study_dir / 'autoresearch.yaml'}\n"
                    f"- Graph: {self._study_dir / 'graph.yaml'}\n"
                    f"- Recipe directory: {self._recipe_dir}\n"
                    "For every propose request, use Read in this agent call on Objective "
                    "(program.md) and Resolved AutoResearch config before reasoning. These "
                    "files are mutable and supersede any resumed-session/cache memory. Then "
                    "read the recipe config paths named by autoresearch.yaml before proposing "
                    "or modifying a trial. Do not propose a patch outside the currently read "
                    "search_space.\n\n"
                    "Recipe-isolation rule: this is an independent experiment. Read and use "
                    "only the listed study paths, files inside the current recipe directory, "
                    "and ESPnet framework code only when needed to understand an API. Do not "
                    "list, search, read, compare against, or reuse information from any other "
                    "recipe under egs3/, including sibling or parent-reachable recipe paths. "
                    "If you accidentally access another recipe, do not use that information; "
                    "report the scope violation and return a failure rather than proposing, "
                    "editing, or accepting an experiment."
                ),
                knowledge="",
                repo_context="",
                trial_history_csv="",
            )
            response = self._delegate.run(request, artifact_dir=artifact_dir)
            if self.session_id is None and response.thread_id:
                self.session_id = response.thread_id
                self._state.set_agent_session_id(self._study_id, self.session_id)
            if response.status == "success" and self.session_id:
                self._state.mark_agent_context_initialized(self._study_id)
            return response
