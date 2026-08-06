"""Recipe-local AutoResearch agent implementations."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from espnet3.autoresearch.agents.interface import AgentRequest, AgentResponse


class UserDefinedCliAgent:
    """Run a user-defined CLI against the prompt and parse JSON output."""

    def __init__(self, config, recipe_dir: Path) -> None:
        self.recipe_dir = Path(recipe_dir).resolve()
        self.command = [str(part) for part in list(getattr(config, "command", []) or [])]
        if not self.command:
            raise ValueError("agent.command must be a non-empty list for UserDefinedCliAgent")
        self.prompt_arg = str(getattr(config, "prompt_arg", "-"))
        self.output_flag = getattr(config, "output_flag", None)
        self.output_mode = str(getattr(config, "output_mode", "last_message_file"))
        self.output_filename = str(getattr(config, "output_filename", "agent_output.json"))
        self.cwd = Path(str(getattr(config, "cwd", self.recipe_dir))).resolve()
        self.env = {
            str(key): str(value)
            for key, value in dict(getattr(config, "env", {}) or {}).items()
        }

    def _prepare_codex_runtime_env(self) -> dict[str, str]:
        """Redirect Codex state into the writable recipe workspace."""
        codex_home = self.recipe_dir / ".codex"
        xdg_config_home = codex_home / ".config"
        xdg_cache_home = codex_home / ".cache"
        home_dir = codex_home / ".home"
        for path in (codex_home, xdg_config_home, xdg_cache_home, home_dir):
            path.mkdir(parents=True, exist_ok=True)

        # Seed the workspace-local Codex home with minimal existing user state
        # so auth/config continue working after redirecting CODEX_HOME.
        user_codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        for file_name in ("auth.json", "config.toml", "installation_id", "version.json"):
            src = user_codex_home / file_name
            dst = codex_home / file_name
            if src.is_file() and not dst.exists():
                shutil.copyfile(src, dst)

        return {
            "CODEX_HOME": str(codex_home),
            "HOME": str(home_dir),
            "XDG_CONFIG_HOME": str(xdg_config_home),
            "XDG_CACHE_HOME": str(xdg_cache_home),
        }

    def _build_prompt(self, request: AgentRequest) -> str:
        return "\n".join(
            [
                f"Task: {request.task}",
                "",
                "Return exactly one JSON object with no markdown fences.",
                "Follow the requested output shape exactly.",
                "",
                "Objective:",
                request.objective.strip(),
                "",
                "Knowledge:",
                request.knowledge.strip(),
                "",
                "Repo Context:",
                request.repo_context.strip(),
                "",
                "Trial History CSV:",
                request.trial_history_csv.strip(),
                "",
                "Latest Metrics:",
                json.dumps(request.latest_metrics, ensure_ascii=False, indent=2),
                "",
                "Latest Logs:",
                json.dumps(request.latest_logs, ensure_ascii=False, indent=2),
                "",
                "Currently Inflight Patches (do NOT propose any of these exactly):",
                json.dumps(request.inflight_patches, ensure_ascii=False, indent=2),
                "",
                "Allowed Actions:",
                ", ".join(request.allowed_actions),
                "",
                "Output Shape:",
                json.dumps(request.output_schema, ensure_ascii=False, indent=2),
            ]
        )

    def _parse_jsonl_events(self, text: str) -> list[dict]:
        events = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        return events

    def _find_usage_objects(self, value) -> list[dict]:
        found = []
        if isinstance(value, dict):
            usage = value.get("usage")
            if isinstance(usage, dict):
                found.append(usage)
            token_keys = {
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
                "reasoning_output_tokens",
            }
            if any(key in value for key in token_keys):
                found.append(
                    {
                        key: value[key]
                        for key in token_keys
                        if isinstance(value.get(key), (int, float))
                    }
                )
            for nested in value.values():
                found.extend(self._find_usage_objects(nested))
        elif isinstance(value, list):
            for item in value:
                found.extend(self._find_usage_objects(item))
        return found

    def _write_usage_logs(
        self,
        artifact_dir: Path,
        request: AgentRequest,
        cmd: list[str],
        stdout_text: str,
    ) -> None:
        events = self._parse_jsonl_events(stdout_text)
        usage_records = []
        for event in events:
            for usage in self._find_usage_objects(event):
                if usage:
                    usage_records.append(usage)

        events_path = artifact_dir / "agent_events.jsonl"
        usage_path = artifact_dir / "agent_usage.json"
        usage_history_path = artifact_dir / "agent_usage_history.jsonl"
        if stdout_text.strip():
            events_path.write_text(stdout_text, encoding="utf-8")

        summary = {
            "task": request.task,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "command": cmd,
            "usage_records": usage_records,
            "last_usage": usage_records[-1] if usage_records else {},
            "event_count": len(events),
        }
        usage_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with usage_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        if artifact_dir is None:
            raise ValueError("artifact_dir is required for UserDefinedCliAgent")
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        prompt = self._build_prompt(request)
        prompt_path = artifact_dir / "agent_prompt.txt"
        stdout_path = artifact_dir / "agent_stdout.log"
        stderr_path = artifact_dir / "agent_stderr.log"
        output_path = artifact_dir / self.output_filename
        prompt_path.write_text(prompt, encoding="utf-8")

        cmd = list(self.command)
        if request.resume_thread_id:
            cmd.extend(["resume", request.resume_thread_id])
        if self.output_flag and not request.resume_thread_id:
            cmd.extend([str(self.output_flag), str(output_path)])
        cmd.append(self.prompt_arg)

        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=self.cwd,
            env={
                **os.environ,
                **self._prepare_codex_runtime_env(),
                **self.env,
            },
            check=False,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        self._write_usage_logs(artifact_dir, request, cmd, proc.stdout)

        # Extract thread_id from thread.started event in JSONL stream
        thread_id: str | None = None
        events = self._parse_jsonl_events(proc.stdout)
        for ev in events:
            if ev.get("type") == "thread.started" and ev.get("thread_id"):
                thread_id = str(ev["thread_id"])
                break

        if proc.returncode != 0:
            return AgentResponse(
                status="failure",
                message=(
                    f"Agent CLI failed with exit code {proc.returncode}: "
                    f"{' '.join(shlex.quote(part) for part in cmd)}"
                ),
                content=proc.stderr.strip() or proc.stdout.strip(),
                thread_id=thread_id,
                artifacts={
                    "prompt": str(prompt_path),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            )

        if request.resume_thread_id:
            # For resume calls, extract the last agent_message text from JSONL stdout
            content = ""
            for ev in reversed(events):
                if ev.get("type") == "item.completed":
                    item = ev.get("item", {})
                    if item.get("type") == "agent_message":
                        content = str(item.get("text", "")).strip()
                        break
            output_path.write_text(content, encoding="utf-8")
        elif self.output_mode == "last_message_file":
            if not output_path.exists():
                return AgentResponse(
                    status="failure",
                    message="Agent CLI did not produce the configured output file.",
                    content="",
                    thread_id=thread_id,
                    artifacts={
                        "prompt": str(prompt_path),
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                    },
                )
            content = output_path.read_text(encoding="utf-8").strip()
        elif self.output_mode == "stdout":
            content = proc.stdout.strip()
            output_path.write_text(content, encoding="utf-8")
        else:
            return AgentResponse(
                status="failure",
                message=f"Unsupported agent.output_mode: {self.output_mode}",
                content="",
                thread_id=thread_id,
            )

        try:
            structured = dict(json.loads(content))
        except json.JSONDecodeError as exc:
            return AgentResponse(
                status="failure",
                message=f"Agent output was not valid JSON: {exc}",
                content=content,
                thread_id=thread_id,
                artifacts={
                    "prompt": str(prompt_path),
                    "output": str(output_path),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            )
        return AgentResponse(
            status="success",
            message="Agent CLI completed.",
            content=content,
            structured=structured,
            thread_id=thread_id,
            artifacts={
                "prompt": str(prompt_path),
                "output": str(output_path),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
        )
