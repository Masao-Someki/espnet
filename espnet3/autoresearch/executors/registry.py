"""Executor registry."""

from __future__ import annotations

from espnet3.autoresearch.executors.local import LocalExecutor
from espnet3.autoresearch.executors.slurm import SlurmConfig, SlurmExecutor


def build_executor(config):
    """Build an executor backend from config."""
    type_ = str(getattr(config, "type", "local"))
    if type_ == "local":
        max_jobs = int(getattr(config, "max_jobs", 4))
        return LocalExecutor(max_jobs=max_jobs)
    if type_ == "slurm":
        return SlurmExecutor(SlurmConfig.from_config(config))
    raise ValueError(f"Unknown executor backend: {type_}")
