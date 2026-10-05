"""AutoResearch configuration schema, loading, and placeholder rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from omegaconf import OmegaConf

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

#: Placeholder names loop.py/runner.py are expected to provide at trial time.
KNOWN_PLACEHOLDERS = frozenset(
    {
        "recipe_dir",
        "study_dir",
        "trial_dir",
        "trial_id",
        "training_config",
        "inference_config",
        "metrics_config",
        "exp_dir",
        "inference_dir",
        "prompt_file",
        "response_file",
    }
)


def render(template: str, context: Mapping[str, Any]) -> str:
    """Render a `{name}` placeholder template (not `str.format`).

    Args:
        template: A string containing zero or more `{name}` placeholders.
        context: Values for every placeholder name that may appear.

    Returns:
        str: `template` with each `{name}` replaced by `str(context[name])`.

    Raises:
        ValueError: If `template` contains a `{name}` not present in
            `context` (so a typo'd or unsupported placeholder fails loudly
            instead of being silently left in place).
    """

    def _sub(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in context:
            raise ValueError(f"Unknown placeholder '{{{name}}}' in {template!r}")
        return str(context[name])

    return _PLACEHOLDER_RE.sub(_sub, template)


def render_list(templates: Sequence[str], context: Mapping[str, Any]) -> List[str]:
    """Apply `render` to every element of `templates`."""
    return [render(t, context) for t in templates]


class ConfigError(ValueError):
    """Raised when `autoresearch.yaml` is missing or has an invalid field."""


def _require(data: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in data or data[key] is None:
        raise ConfigError(f"{section}.{key} is required")
    return data[key]


@dataclass
class RuntimeKeyMap:
    """Dotted key -> placeholder-template value, per role, applied pre-resolve."""

    training: Dict[str, str] = field(default_factory=dict)
    inference: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeKeyMap":
        """Build a `RuntimeKeyMap` from a plain dict (missing roles default empty)."""
        return cls(
            training=dict(data.get("training") or {}),
            inference=dict(data.get("inference") or {}),
            metrics=dict(data.get("metrics") or {}),
        )


@dataclass
class RecipeConfig:
    """Paths to the recipe's role configs, plus per-trial runtime key overrides."""

    training_config: str
    inference_config: str
    metrics_config: str
    runtime_keys: RuntimeKeyMap

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecipeConfig":
        """Build a `RecipeConfig` from a plain dict, requiring the 3 config paths."""
        return cls(
            training_config=_require(data, "training_config", "recipe"),
            inference_config=_require(data, "inference_config", "recipe"),
            metrics_config=_require(data, "metrics_config", "recipe"),
            runtime_keys=RuntimeKeyMap.from_dict(data.get("runtime_keys") or {}),
        )


@dataclass
class TrialConfig:
    """How to run one trial: argv commands, working dir, timeout, and env."""

    workdir: str
    commands: List[List[str]]
    timeout_sec: int
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrialConfig":
        """Build a `TrialConfig` from a plain dict, validating `commands`."""
        commands = _require(data, "commands", "trial")
        if not isinstance(commands, list) or not commands:
            raise ConfigError("trial.commands must be a non-empty list of argv lists")
        for cmd in commands:
            if not isinstance(cmd, list) or not cmd:
                raise ConfigError("trial.commands entries must be non-empty lists")
        return cls(
            workdir=data.get("workdir", "{recipe_dir}"),
            commands=[[str(arg) for arg in cmd] for cmd in commands],
            timeout_sec=int(data.get("timeout_sec", 3600)),
            env=dict(data.get("env") or {}),
        )


@dataclass
class MetricSourceEntry:
    """One `metric.source` candidate: a file path plus a JSON key or pattern."""

    path: str
    key: Optional[str] = None
    pattern: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricSourceEntry":
        """Build a `MetricSourceEntry`, requiring `path` and `key` or `pattern`."""
        path = _require(data, "path", "metric.source")
        key = data.get("key")
        pattern = data.get("pattern")
        if key is None and pattern is None:
            raise ConfigError("metric.source entries need 'key' or 'pattern'")
        return cls(path=path, key=key, pattern=pattern)


@dataclass
class MetricConfig:
    """The trial's objective: its name, optimization direction, and sources."""

    name: str
    mode: str
    source: List[MetricSourceEntry]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricConfig":
        """Build a `MetricConfig` from a plain dict, validating `mode` and `source`."""
        mode = _require(data, "mode", "metric")
        if mode not in ("min", "max"):
            raise ConfigError(f"metric.mode must be 'min' or 'max', got {mode!r}")
        source = _require(data, "source", "metric")
        if not isinstance(source, list) or not source:
            raise ConfigError("metric.source must be a non-empty list")
        return cls(
            name=_require(data, "name", "metric"),
            mode=mode,
            source=[MetricSourceEntry.from_dict(entry) for entry in source],
        )


@dataclass
class BudgetConfig:
    """Stop conditions for the trial loop."""

    max_trials: int
    max_failures: int
    no_improve_stop: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BudgetConfig":
        """Build a `BudgetConfig` from a plain dict (all fields default)."""
        return cls(
            max_trials=int(data.get("max_trials", 50)),
            max_failures=int(data.get("max_failures", 5)),
            no_improve_stop=int(data.get("no_improve_stop", 10)),
        )


@dataclass
class SearchSpaceConfig:
    """Glob patterns constraining which dotted keys an agent's patch may touch."""

    allowed_keys: List[str] = field(default_factory=list)
    denied_keys: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchSpaceConfig":
        """Build a `SearchSpaceConfig` from a plain dict (both lists default empty)."""
        return cls(
            allowed_keys=list(data.get("allowed_keys") or []),
            denied_keys=list(data.get("denied_keys") or []),
        )


@dataclass
class AgentConfig:
    """How to invoke the proposing agent: its type, launch command, and I/O format."""

    type: str = "command"
    command: List[str] = field(default_factory=list)
    prompt_via: str = "stdin"
    response_format: str = "json"
    timeout_sec: int = 1800
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        """Build an `AgentConfig` from a plain dict, validating `type`/`prompt_via`."""
        agent_type = data.get("type", "command")
        if agent_type not in ("command", "file"):
            raise ConfigError(
                f"agent.type must be 'command' or 'file', got {agent_type!r}"
            )
        if agent_type == "command" and not data.get("command"):
            raise ConfigError("agent.command is required when agent.type == 'command'")
        prompt_via = data.get("prompt_via", "stdin")
        if prompt_via not in ("stdin", "file"):
            raise ConfigError(
                f"agent.prompt_via must be 'stdin' or 'file', got {prompt_via!r}"
            )
        return cls(
            type=agent_type,
            command=[str(arg) for arg in (data.get("command") or [])],
            prompt_via=prompt_via,
            response_format=data.get("response_format", "json"),
            timeout_sec=int(data.get("timeout_sec", 1800)),
            env=dict(data.get("env") or {}),
        )


@dataclass
class EditConfig:
    """How an agent is allowed to edit files during a trial (PR2)."""

    mode: str = "direct"
    allowlist: List[str] = field(default_factory=list)
    on_violation: str = "fail"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EditConfig":
        """Build an `EditConfig` from a plain dict, validating `mode`/`allowlist`."""
        mode = data.get("mode", "direct")
        if mode not in ("direct", "allowlist"):
            raise ConfigError(
                f"edit.mode must be 'direct' or 'allowlist', got {mode!r}"
            )
        if mode == "allowlist" and not data.get("allowlist"):
            raise ConfigError(
                "edit.allowlist is required when edit.mode == 'allowlist'"
            )
        on_violation = data.get("on_violation", "fail")
        if on_violation != "fail":
            raise ConfigError("edit.on_violation only supports 'fail' for now")
        return cls(
            mode=mode,
            allowlist=list(data.get("allowlist") or []),
            on_violation=on_violation,
        )


@dataclass
class AutoResearchConfig:
    """Top-level, validated `autoresearch.yaml` configuration."""

    study_name: str
    study_dir: str
    objective_file: str
    recipe: RecipeConfig
    trial: TrialConfig
    metric: MetricConfig
    budget: BudgetConfig
    search_space: SearchSpaceConfig
    agent: AgentConfig
    edit: EditConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AutoResearchConfig":
        """Build an `AutoResearchConfig` from a plain dict, validating every section."""
        return cls(
            study_name=_require(data, "study_name", "autoresearch"),
            study_dir=_require(data, "study_dir", "autoresearch"),
            objective_file=data.get("objective_file", "program.md"),
            recipe=RecipeConfig.from_dict(_require(data, "recipe", "autoresearch")),
            trial=TrialConfig.from_dict(_require(data, "trial", "autoresearch")),
            metric=MetricConfig.from_dict(_require(data, "metric", "autoresearch")),
            budget=BudgetConfig.from_dict(data.get("budget") or {}),
            search_space=SearchSpaceConfig.from_dict(data.get("search_space") or {}),
            agent=AgentConfig.from_dict(_require(data, "agent", "autoresearch")),
            edit=EditConfig.from_dict(data.get("edit") or {}),
        )


def load_config(path: Path) -> AutoResearchConfig:
    """Load and validate an `autoresearch.yaml` (resolving `${...}` interpolations).

    Accepts either a file with a top-level `autoresearch:` key (the TEMPLATE
    convention) or one whose root is already the config body.

    Args:
        path: Path to the YAML file.

    Returns:
        AutoResearchConfig: The parsed, validated configuration.

    Raises:
        ConfigError: If a required field is missing or has an invalid value.
    """
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at its root")
    data = raw.get("autoresearch", raw)
    return AutoResearchConfig.from_dict(data)
