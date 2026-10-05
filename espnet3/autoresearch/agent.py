"""Agent request/response types and the two agent implementations.

An agent takes an :class:`AgentRequest` describing the current trial and
returns an :class:`AgentResponse` proposing a config patch. This module owns
only the agent side of that exchange: how a proposal is requested and
parsed. It is deliberately independent of ``config.py``'s
``AutoResearchConfig`` dataclass -- callers (``loop.py``) pass plain values
(an argv list, a timeout, a placeholder mapping, an :class:`AgentRequest`)
instead of a config object, so this module has no import-time dependency on
the rest of the ``autoresearch`` package.

Two implementations:

- :class:`CommandAgent`: runs a CLI agent (e.g. ``codex exec``) as a
  subprocess, built from an argv template with ``{placeholder}``
  substitution.
- :class:`FileAgent`: a human-in-the-loop handoff. Writes the rendered
  request to a file and reads a response file back (ported from ATLAS's
  ``FileAgentClient``).

Both share :func:`render_prompt` for the request's human-readable form, and
raise :class:`AgentResponseError` (reason ``"agent_response"``) when a
response cannot be read.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

#: `AgentRequest.context` keys :func:`render_prompt` looks for. All optional;
#: a missing key renders as an explicit "(none)" rather than being omitted,
#: so the agent is not left guessing whether a section was left out on
#: purpose.
_CONTEXT_KEYS = (
    "edit_mode",
    "edit_allowlist",
    "base_config",
    "search_space",
    "trial_history_csv",
    "best_patch",
    "best_score",
    "recent_failure_log",
)

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass
class AgentRequest:
    """What a trial asks an agent to propose.

    Attributes:
        task: Short label for what is being asked (e.g. `"propose_patch"`).
        objective: The study's objective (`program.md`'s contents).
        context: Everything else :func:`render_prompt` folds into the
            prompt, keyed by the names in `_CONTEXT_KEYS`
            (`edit_mode`, `edit_allowlist`, `base_config`, `search_space`,
            `trial_history_csv`, `best_patch`, `best_score`,
            `recent_failure_log`). Unknown keys are ignored by
            `render_prompt`; missing keys render as "(none)".
        allowed_actions: Human-readable list of what the agent may do.
        output_schema: JSON-serializable shape the agent's response should
            follow (included in the prompt, not enforced here).
    """

    task: str
    objective: str
    context: Mapping[str, Any] = field(default_factory=dict)
    allowed_actions: Sequence[str] = field(default_factory=list)
    output_schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """A proposed config patch and the agent's reasoning for it.

    Attributes:
        rationale: Why the agent proposed this patch.
        config_patch: Dotted-key patch (see `patch.py`'s
            `apply_dotted_patch`), e.g. `{"trainer.lr": 0.001}`.
        expected_effect: What the agent expects this patch to change.
        risk: What could go wrong, in the agent's own words.
    """

    rationale: str
    config_patch: Mapping[str, Any]
    expected_effect: str = ""
    risk: str = ""


class AgentResponseError(Exception):
    """An agent's response could not be obtained, read, or parsed.

    `reason` is always `"agent_response"`, so callers (`loop.py`) can record
    it verbatim as the trial's failure reason without inspecting the
    message.
    """

    reason = "agent_response"


class AgentWaitingError(Exception):
    """`FileAgent` found no response file and was told not to wait.

    Not a trial failure: the response simply has not been written yet.
    Callers should pause the study run here, not fail the trial.
    """


def render_placeholders(parts: Sequence[str], values: Mapping[str, str]) -> list[str]:
    """Substitute `{name}` placeholders in each of `parts`.

    Unlike `str.format`, a literal `{`/`}` that is not a recognized
    `{name}` placeholder is left alone (so argv entries containing JSON or
    shell braces are not mangled), but a `{name}` that *is* well-formed
    and absent from `values` raises `ValueError` naming it -- a silently
    unsubstituted placeholder reaching a subprocess argv is always a bug.

    Args:
        parts: Argv entries to render (e.g. `command`, before substitution).
        values: Placeholder name -> replacement value.

    Returns:
        list[str]: `parts` with every `{name}` replaced.

    Raises:
        ValueError: If a `{name}` in `parts` is not a key of `values`.
    """
    rendered = []
    for part in parts:
        missing = [name for name in _PLACEHOLDER_RE.findall(part) if name not in values]
        if missing:
            raise ValueError(
                f"Unknown placeholder(s) {missing} in {part!r}; "
                f"known placeholders: {sorted(values)}"
            )
        rendered.append(_PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]), part))
    return rendered


def render_prompt(request: AgentRequest) -> str:
    """Render `request` into the fixed-section prompt both agents send.

    Sections, in order: Objective, Edit Rules, Base Training Config, Search
    Space, Trial History, Best Patch & Score, Recent Failure Log, Allowed
    Actions, Output Schema. A section whose `context` key is absent renders
    as `(none)` rather than being dropped.
    """
    ctx = request.context

    edit_mode = ctx.get("edit_mode") or "(none)"
    edit_allowlist = ctx.get("edit_allowlist") or []
    edit_rules = (
        f"mode: {edit_mode}"
        + ("\nallowlist:\n" + "\n".join(f"  - {p}" for p in edit_allowlist))
        if edit_allowlist
        else f"mode: {edit_mode}"
    )

    best_patch = ctx.get("best_patch")
    best_score = ctx.get("best_score")
    if best_patch is None and best_score is None:
        best_section = "(none)"
    else:
        best_section = (
            f"score: {best_score}\n"
            f"patch:\n```json\n{json.dumps(best_patch or {}, indent=2)}\n```"
        )

    return "\n".join(
        [
            f"# AutoResearch Agent Request: {request.task}",
            "",
            "## Objective",
            request.objective.strip(),
            "",
            "## Edit Rules",
            edit_rules,
            "",
            "## Base Training Config",
            "```yaml",
            str(ctx.get("base_config") or "(none)").strip(),
            "```",
            "",
            "## Search Space",
            "```json",
            json.dumps(ctx.get("search_space") or {}, indent=2),
            "```",
            "",
            "## Trial History",
            "```csv",
            str(ctx.get("trial_history_csv") or "(none)").strip(),
            "```",
            "",
            "## Best Patch & Score",
            best_section,
            "",
            "## Recent Failure Log",
            "```",
            str(ctx.get("recent_failure_log") or "(none)").strip(),
            "```",
            "",
            "## Allowed Actions",
            ", ".join(request.allowed_actions) or "(none)",
            "",
            "## Output Schema",
            "```json",
            json.dumps(request.output_schema, indent=2),
            "```",
        ]
    )


def _load_structured(path: Path, response_format: str) -> Mapping[str, Any]:
    """Read and parse `path` as JSON or YAML.

    Raises `AgentResponseError` (not the underlying parser's exception
    type) on any failure.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise AgentResponseError(f"could not read response file {path}: {e}") from e

    try:
        if response_format == "json":
            data = json.loads(text)
        elif response_format == "yaml":
            data = yaml.safe_load(text)
        else:
            raise AgentResponseError(
                f"unsupported response_format {response_format!r}; "
                "expected 'json' or 'yaml'"
            )
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise AgentResponseError(f"could not parse response file {path}: {e}") from e

    if not isinstance(data, Mapping):
        raise AgentResponseError(
            f"response file {path} did not contain a JSON/YAML object "
            f"(got {type(data).__name__})"
        )
    return data


def _response_from_mapping(data: Mapping[str, Any]) -> AgentResponse:
    """Build an `AgentResponse` from a parsed response mapping.

    `rationale` and `config_patch` are required; a response missing either
    is treated the same as an unparseable one (`AgentResponseError`), since
    a patch-less proposal cannot drive a trial.
    """
    if "config_patch" not in data:
        raise AgentResponseError("response is missing required key 'config_patch'")
    return AgentResponse(
        rationale=str(data.get("rationale", "")),
        config_patch=dict(data["config_patch"] or {}),
        expected_effect=str(data.get("expected_effect", "")),
        risk=str(data.get("risk", "")),
    )


class CommandAgent:
    """Runs a CLI agent (e.g. `codex exec`) as a subprocess for one proposal.

    `command` is an argv template (e.g.
    `["codex", "exec", "-C", "{recipe_dir}", "--output-last-message",
    "{response_file}", "-"]`); `propose()` fills in `{placeholder}`s (the
    caller-supplied ones plus `{prompt_file}`/`{response_file}`, which this
    class owns) and runs it. The prompt is sent over stdin or via
    `{prompt_file}`, per `prompt_via`. The response is read back from
    `{response_file}` as JSON or YAML, per `response_format`.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        prompt_via: str = "stdin",
        response_format: str = "json",
        timeout_sec: float = 1800.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Build a `CommandAgent`; see the class docstring for the fields."""
        if not command:
            raise ValueError("command must be a non-empty argv list")
        if prompt_via not in ("stdin", "file"):
            raise ValueError(
                f"prompt_via must be 'stdin' or 'file', got {prompt_via!r}"
            )
        if response_format not in ("json", "yaml"):
            raise ValueError(
                f"response_format must be 'json' or 'yaml', got {response_format!r}"
            )
        self.command = [str(part) for part in command]
        self.prompt_via = prompt_via
        self.response_format = response_format
        self.timeout_sec = timeout_sec
        self.env = dict(env or {})

    def propose(
        self,
        request: AgentRequest,
        *,
        artifact_dir: str | Path,
        placeholders: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> AgentResponse:
        """Run the agent once and return its proposal.

        Args:
            request: What to ask the agent.
            artifact_dir: Directory for `prompt.md`, `response.<ext>`,
                `stdout.log`, `stderr.log` (created if missing).
            placeholders: Extra `{name}` values for `command` (e.g.
                `recipe_dir`, `trial_id`). `prompt_file`/`response_file` are
                added automatically and override any caller-supplied value
                for those two names.
            cwd: Working directory for the subprocess. Defaults to the
                caller's own cwd (unchanged) when `None`.

        Returns:
            AgentResponse: The parsed proposal.

        Raises:
            AgentResponseError: If the subprocess fails, times out, or its
                response file is missing, unreadable, or malformed.
        """
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = render_prompt(request)
        prompt_file = artifact_dir / "prompt.md"
        prompt_file.write_text(prompt_text, encoding="utf-8")
        response_file = artifact_dir / f"response.{self.response_format}"
        response_file.unlink(missing_ok=True)

        values: MutableMapping[str, str] = dict(placeholders or {})
        values["prompt_file"] = str(prompt_file)
        values["response_file"] = str(response_file)
        argv = render_placeholders(self.command, values)

        stdin_data = prompt_text if self.prompt_via == "stdin" else None
        stdout_text = stderr_text = ""
        try:
            result = subprocess.run(
                argv,
                input=stdin_data,
                cwd=cwd,
                env=self.env or None,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            stdout_text, stderr_text = result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            stdout_text = e.stdout or ""
            stderr_text = e.stderr or ""
        finally:
            (artifact_dir / "stdout.log").write_text(stdout_text, encoding="utf-8")
            (artifact_dir / "stderr.log").write_text(stderr_text, encoding="utf-8")

        if not response_file.exists():
            raise AgentResponseError(
                f"agent command produced no response file: {response_file}"
            )
        data = _load_structured(response_file, self.response_format)
        return _response_from_mapping(data)


class FileAgent:
    """Human-in-the-loop agent: writes a request file, reads a response file.

    Ported from ATLAS's `FileAgentClient` onto the new `AgentRequest`/
    `AgentResponse` types.
    """

    def __init__(
        self,
        *,
        request_filename: str = "agent_request.md",
        response_filename: str = "agent_response.yaml",
        wait: bool = False,
        poll_interval_sec: float = 5.0,
        timeout_sec: float = 3600.0,
    ) -> None:
        """Build a `FileAgent`; see the class docstring for the fields."""
        self.request_filename = request_filename
        self.response_filename = response_filename
        self.wait = wait
        self.poll_interval_sec = poll_interval_sec
        self.timeout_sec = timeout_sec

    def propose(
        self, request: AgentRequest, *, artifact_dir: str | Path
    ) -> AgentResponse:
        """Write the rendered request, then return or wait for a response.

        Args:
            request: What to ask the (human) agent.
            artifact_dir: Directory for the request/response files (created
                if missing).

        Returns:
            AgentResponse: The parsed proposal, once available.

        Raises:
            AgentWaitingError: No response file yet, and `wait=False`.
            AgentResponseError: The response file is unreadable/malformed,
                or (with `wait=True`) `timeout_sec` elapsed before one
                appeared.
        """
        import time

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_path = artifact_dir / self.request_filename
        response_path = artifact_dir / self.response_filename
        request_path.write_text(render_prompt(request), encoding="utf-8")

        if response_path.exists():
            return _response_from_mapping(_load_structured(response_path, "yaml"))
        if not self.wait:
            raise AgentWaitingError(f"waiting for agent response: {response_path}")

        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            if response_path.exists():
                return _response_from_mapping(_load_structured(response_path, "yaml"))
            time.sleep(self.poll_interval_sec)
        raise AgentResponseError(
            f"timed out waiting for agent response: {response_path}"
        )
