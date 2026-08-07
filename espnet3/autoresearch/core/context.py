"""Stage execution context."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from espnet3.autoresearch.core.trial import Trial

if TYPE_CHECKING:
    from espnet3.autoresearch.core.graph import ResearchGraph
    from espnet3.autoresearch.core.scheduler import AutoResearchGraphRuntime
    from espnet3.autoresearch.core.state import StateStore


@dataclass
class StageContext:
    """Runtime context shared with a stage."""

    study_dir: Path
    recipe_dir: Path
    node_name: str
    run_id: str
    attempt_id: str
    graph: "ResearchGraph"
    config: DictConfig
    state: "StateStore"
    executor: Any
    agent: Any
    search: Any
    logger: logging.Logger
    scheduler: "AutoResearchGraphRuntime"
    current_trial: Trial | None = None

    def trial_dir(self) -> Path:
        assert self.current_trial is not None
        return self.study_dir / "trials" / self.current_trial.trial_id

    def artifact_dir(self, *parts: str) -> Path:
        return self.study_dir / "node_runs" / self.run_id / Path(*parts)
