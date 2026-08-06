"""Filesystem-backed agent handoff."""

from __future__ import annotations

import time
from pathlib import Path

from espnet3.autoresearch.agents.interface import AgentRequest, AgentResponse
from espnet3.autoresearch.core.serialization import dumps_json, load_yaml, save_yaml


class FileAgentClient:
    """Write request files and poll for a response file."""

    def __init__(
        self,
        request_filename: str = "agent_request.md",
        response_filename: str = "agent_response.yaml",
        wait: bool = False,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ) -> None:
        self.request_filename = request_filename
        self.response_filename = response_filename
        self.wait = wait
        self.poll_interval = poll_interval
        self.timeout = timeout

    def _render_request(self, request: AgentRequest) -> str:
        return "\n".join(
            [
                f"# AutoResearch Agent Request: {request.task}",
                "",
                "## Objective",
                request.objective.strip(),
                "",
                "## Knowledge",
                request.knowledge.strip(),
                "",
                "## Repo Context",
                request.repo_context.strip(),
                "",
                "## Trial History CSV",
                "```csv",
                request.trial_history_csv.strip(),
                "```",
                "",
                "## Latest Metrics",
                "```json",
                dumps_json(request.latest_metrics),
                "```",
                "",
                "## Latest Logs",
                "```json",
                dumps_json(request.latest_logs),
                "```",
                "",
                "## Currently Inflight Patches (do NOT propose any of these exactly)",
                "```json",
                dumps_json(request.inflight_patches),
                "```",
                "",
                "## Allowed Actions",
                ", ".join(request.allowed_actions),
                "",
                "## Output Schema",
                "```json",
                dumps_json(request.output_schema),
                "```",
                "",
            ]
        )

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        if artifact_dir is None:
            raise ValueError("artifact_dir is required for FileAgentClient")
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_path = artifact_dir / self.request_filename
        request_json_path = artifact_dir / "agent_request.json"
        response_path = artifact_dir / self.response_filename
        request_path.write_text(self._render_request(request), encoding="utf-8")
        save_yaml(request_json_path, request.__dict__)

        if response_path.exists():
            data = load_yaml(response_path)
            return AgentResponse(
                status=str(data.get("status", "success")),
                message=str(data.get("message", "")),
                content=str(data.get("content", "")),
                structured=dict(data.get("structured", {}) or data),
                artifacts=dict(data.get("artifacts", {}) or {}),
            )

        if not self.wait:
            return AgentResponse(
                status="failure",
                message=f"Waiting for agent response: {response_path}",
                content="",
            )

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if response_path.exists():
                data = load_yaml(response_path)
                return AgentResponse(
                    status=str(data.get("status", "success")),
                    message=str(data.get("message", "")),
                    content=str(data.get("content", "")),
                    structured=dict(data.get("structured", {}) or data),
                    artifacts=dict(data.get("artifacts", {}) or {}),
                )
            time.sleep(self.poll_interval)
        return AgentResponse(
            status="failure",
            message=f"Timed out waiting for agent response: {response_path}",
            content="",
        )
