"""Graph loading and routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from espnet3.autoresearch.core.errors import ConfigValidationError, GraphRoutingError
from espnet3.autoresearch.core.loader import load_stage_class
from espnet3.autoresearch.core.serialization import load_yaml
from espnet3.autoresearch.executors.base import Resources


@dataclass
class NodeConfig:
    """Per-node graph config."""

    name: str
    target: str
    executor: str
    resources: Resources | None
    on: dict[str, str] = field(default_factory=dict)
    terminal: bool = False


@dataclass
class ResearchGraph:
    """Static stage graph."""

    name: str
    version: int
    start: str
    nodes: dict[str, NodeConfig]

    def route(self, node_name: str, status: str) -> str | None:
        node = self.nodes[node_name]
        if node.terminal:
            return None
        next_node = node.on.get(status)
        if next_node is None:
            raise GraphRoutingError(
                f"Node '{node_name}' has no routing for status '{status}'. "
                f"Defined: {sorted(node.on.keys())}"
            )
        return next_node


def load_graph(path: Path, recipe_dir: Path | None = None) -> ResearchGraph:
    """Load a graph YAML and validate stage imports."""
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Graph YAML must be a mapping: {path}")
    nodes_raw = raw.get("nodes")
    if not nodes_raw:
        raise ConfigValidationError("Graph must define non-empty `nodes`.")
    nodes: dict[str, NodeConfig] = {}
    for name, cfg in nodes_raw.items():
        target = str(cfg.get("target", ""))
        load_stage_class(target, recipe_dir=recipe_dir)
        executor = str(cfg.get("executor", "local"))
        resources_cfg = cfg.get("resources")
        resources = Resources.from_config(resources_cfg) if resources_cfg else None
        on = cfg.get("on", {})
        terminal = bool(cfg.get("terminal", False))
        nodes[name] = NodeConfig(
            name=name,
            target=target,
            executor=executor,
            resources=resources,
            on=on,
            terminal=terminal,
        )
    start = raw.get("start")
    if start not in nodes:
        raise ConfigValidationError(f"Graph start node '{start}' is not defined.")
    graph_name = str(raw.get("name", path.stem))
    version = int(raw.get("version", 1))
    return ResearchGraph(name=graph_name, version=version, start=start, nodes=nodes)
