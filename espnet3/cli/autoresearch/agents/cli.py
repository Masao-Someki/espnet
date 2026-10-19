"""Claude Code and Codex command-line agent."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from espnet3.cli.autoresearch.agents.interface import AgentRequest, AgentResponse
from espnet3.cli.autoresearch.core.models import AgentConfig, ClaudeConfig, CodexConfig


def _toml_value(value: Any) -> str:
    """Serialize a value accepted by Codex's ``-c key=value`` option."""
    if value is None:
        raise ValueError("Codex config overrides cannot contain null values")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = ", ".join(
            f"{key} = {_toml_value(item)}" for key, item in value.items()
        )
        return "{" + fields + "}"
    raise TypeError(f"Unsupported Codex config value: {type(value).__name__}")


class CliAgent:
    """Run Codex or Claude Code and return its response."""

    def __init__(self, config, recipe_dir: Path) -> None:
        if isinstance(config, AgentConfig):
            self.config = config
        else:
            self.config = AgentConfig.from_config(config)
        self.recipe_dir = Path(recipe_dir).resolve()
        self.cwd = Path(self.config.cwd or self.recipe_dir).resolve()

    def _build_prompt(self, request: AgentRequest) -> str:
        parts = [
            f"Task: {request.task}",
            "",
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
            "Additional Context (including log paths when relevant):",
            json.dumps(request.context_files, ensure_ascii=False, indent=2),
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
        if request.output_schema:
            parts.insert(2, "Return exactly one JSON object with no markdown fences.")
        return "\n".join(parts)

    def _codex_env(self) -> dict[str, str]:
        """Give Codex a workspace-local home while retaining user credentials."""
        codex_home = self.recipe_dir / ".codex"
        xdg_config_home = codex_home / ".config"
        xdg_cache_home = codex_home / ".cache"
        home_dir = codex_home / ".home"
        for path in (codex_home, xdg_config_home, xdg_cache_home, home_dir):
            path.mkdir(parents=True, exist_ok=True)

        user_codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        for name in ("auth.json", "config.toml", "installation_id", "version.json"):
            source = user_codex_home / name
            destination = codex_home / name
            if source.is_file() and not destination.exists():
                shutil.copyfile(source, destination)
        return {
            "CODEX_HOME": str(codex_home),
            "HOME": str(home_dir),
            "XDG_CONFIG_HOME": str(xdg_config_home),
            "XDG_CACHE_HOME": str(xdg_cache_home),
        }

    def _build_codex_command(
        self, request: AgentRequest, output_path: Path
    ) -> list[str]:
        options = self.config.codex
        if request.resume_thread_id:
            command = [
                options.executable,
                "exec",
                "resume",
                request.resume_thread_id,
                "--json",
            ]
        else:
            command = [options.executable, "exec", "--json", "-C", str(self.cwd)]
        if options.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if options.sandbox:
            if request.resume_thread_id:
                if options.sandbox == "danger-full-access":
                    command.append("--dangerously-bypass-approvals-and-sandbox")
                else:
                    command.extend(
                        ["-c", f"sandbox_mode={_toml_value(options.sandbox)}"]
                    )
            else:
                command.extend(["--sandbox", options.sandbox])
        if options.model:
            command.extend(["--model", options.model])
        for key, value in options.config.items():
            command.extend(["-c", f"{key}={_toml_value(value)}"])
        command.extend(options.args)
        command.extend(["--output-last-message", str(output_path)])
        command.append("-")
        return command

    def _build_claude_command(self, request: AgentRequest) -> list[str]:
        options = self.config.claude
        command = [
            options.executable,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            options.permission_mode,
        ]
        if options.model:
            command.extend(["--model", options.model])
        if options.effort:
            command.extend(["--effort", options.effort])
        if options.max_turns is not None:
            command.extend(["--max-turns", str(options.max_turns)])
        if options.allowed_tools:
            command.extend(["--allowedTools", *options.allowed_tools])
        if options.disallowed_tools:
            command.extend(["--disallowedTools", *options.disallowed_tools])
        command.extend(options.args)
        if request.resume_thread_id:
            command.extend(["--resume", request.resume_thread_id])
        return command

    @staticmethod
    def _events(stdout: str) -> list[dict[str, Any]]:
        events = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @staticmethod
    def _extract_codex_content(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    return str(item.get("text", "")).strip()
        return ""

    @staticmethod
    def _extract_claude_content(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            if event.get("type") == "result":
                return str(event.get("result", "")).strip()
        text = []
        for event in events:
            if event.get("type") != "assistant":
                continue
            message = event.get("message", {})
            for item in message.get("content", []):
                if item.get("type") == "text":
                    text.append(str(item.get("text", "")))
        return "\n".join(text).strip()

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        if artifact_dir is None:
            raise ValueError("artifact_dir is required for CliAgent")
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prompt = self._build_prompt(request)
        prompt_path = artifact_dir / "agent_prompt.txt"
        output_path = artifact_dir / "agent_output.txt"
        stdout_path = artifact_dir / "agent_stdout.log"
        stderr_path = artifact_dir / "agent_stderr.log"
        events_path = artifact_dir / "agent_events.jsonl"
        prompt_path.write_text(prompt, encoding="utf-8")

        if self.config.provider == "codex":
            command = self._build_codex_command(request, output_path)
            extra_env = self._codex_env()
        else:
            command = self._build_claude_command(request)
            extra_env = {}
        try:
            proc = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=self.cwd,
                env={**os.environ, **extra_env, **self.config.env},
                timeout=self.config.timeout_sec,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            message = (
                f"Agent executable was not found: {command[0]}"
                if isinstance(exc, FileNotFoundError)
                else f"Agent CLI timed out after {self.config.timeout_sec} seconds."
            )
            return AgentResponse(status="failure", message=message, content="")

        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.stdout.strip():
            events_path.write_text(proc.stdout, encoding="utf-8")
        events = self._events(proc.stdout)
        if self.config.provider == "codex":
            self._write_codex_usage(artifact_dir, request, command, events)
        thread_id = next(
            (
                str(event[key])
                for event in events
                for key in ("thread_id", "session_id")
                if event.get(key)
            ),
            None,
        )
        if proc.returncode != 0:
            return AgentResponse(
                status="failure",
                message=(
                    f"Agent CLI failed with exit code {proc.returncode}: "
                    f"{' '.join(shlex.quote(part) for part in command)}"
                ),
                content=proc.stderr.strip() or proc.stdout.strip(),
                thread_id=thread_id,
                artifacts={
                    "prompt": str(prompt_path),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            )

        if self.config.provider == "codex":
            if output_path.exists():
                content = output_path.read_text(encoding="utf-8").strip()
            else:
                content = self._extract_codex_content(events)
        else:
            content = self._extract_claude_content(events)
        output_path.write_text(content, encoding="utf-8")
        try:
            structured = json.loads(content)
            if not isinstance(structured, dict):
                structured = {}
        except json.JSONDecodeError:
            structured = {}
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
                "events": str(events_path),
            },
        )

    @staticmethod
    def _find_usage_objects(value) -> list[dict[str, Any]]:
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
                found.extend(CliAgent._find_usage_objects(nested))
        elif isinstance(value, list):
            for item in value:
                found.extend(CliAgent._find_usage_objects(item))
        return found

    @staticmethod
    def _write_codex_usage(
        artifact_dir: Path,
        request: AgentRequest,
        command: list[str],
        events: list[dict[str, Any]],
    ) -> None:
        usage_records = [
            usage
            for event in events
            for usage in CliAgent._find_usage_objects(event)
            if usage
        ]
        summary = {
            "task": request.task,
            "timestamp": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "command": command,
            "usage_records": usage_records,
            "last_usage": usage_records[-1] if usage_records else {},
            "event_count": len(events),
        }
        (artifact_dir / "agent_usage.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (artifact_dir / "agent_usage_history.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
