"""Slurm status helpers."""

from __future__ import annotations

from espnet3.autoresearch.executors.base import JobState

SLURM_STATE_MAP: dict[str, JobState] = {
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "TIMEOUT": "timeout",
    "CANCELLED": "cancelled",
    "PREEMPTED": "cancelled",
}


def map_slurm_state(value: str | None) -> JobState:
    """Map a raw Slurm state string to a generic state."""
    if not value:
        return "unknown"
    return SLURM_STATE_MAP.get(str(value).strip().upper(), "unknown")
