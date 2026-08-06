"""Executor interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass
class Resources:
    """Requested resources for a job."""

    gpus: int = 0
    cpus: int = 1
    nodes: int = 1
    mem: str | None = None
    time: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    reservation: str | None = None
    nodelist: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config) -> "Resources":
        if config is None:
            return cls()
        return cls(
            gpus=int(getattr(config, "gpus", 0) or 0),
            cpus=int(getattr(config, "cpus", 1) or 1),
            nodes=int(getattr(config, "nodes", 1) or 1),
            mem=getattr(config, "mem", None),
            time=getattr(config, "time", None),
            partition=getattr(config, "partition", None),
            account=getattr(config, "account", None),
            qos=getattr(config, "qos", None),
            constraint=getattr(config, "constraint", None),
            reservation=getattr(config, "reservation", None),
            nodelist=getattr(config, "nodelist", None),
            extra=dict(getattr(config, "extra", {}) or {}),
        )

    def merged_with(self, override: "Resources | None") -> "Resources":
        if override is None:
            return Resources(
                gpus=self.gpus,
                cpus=self.cpus,
                nodes=self.nodes,
                mem=self.mem,
                time=self.time,
                partition=self.partition,
                account=self.account,
                qos=self.qos,
                constraint=self.constraint,
                reservation=self.reservation,
                nodelist=self.nodelist,
                extra=dict(self.extra),
            )
        return Resources(
            gpus=override.gpus if override.gpus != 0 else self.gpus,
            cpus=override.cpus if override.cpus != 1 else self.cpus,
            nodes=override.nodes if override.nodes != 1 else self.nodes,
            mem=override.mem or self.mem,
            time=override.time or self.time,
            partition=override.partition or self.partition,
            account=override.account or self.account,
            qos=override.qos or self.qos,
            constraint=override.constraint or self.constraint,
            reservation=override.reservation or self.reservation,
            nodelist=override.nodelist or self.nodelist,
            extra={**self.extra, **override.extra},
        )


@dataclass
class JobSpec:
    """Complete job submission payload."""

    name: str
    command: list[str]
    workdir: Path
    stdout: Path
    stderr: Path
    resources: Resources
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobHandle:
    """Returned by an executor on submit."""

    backend: str
    external_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


JobState = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "timeout",
    "unknown",
]


class ExecutionBackend(Protocol):
    """Executor protocol."""

    def submit(self, spec: JobSpec) -> JobHandle:
        """Submit a job."""

    def status(self, handle: JobHandle) -> JobState:
        """Return job state."""

    def cancel(self, handle: JobHandle) -> None:
        """Cancel a job."""
