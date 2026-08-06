"""Local subprocess executor."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from espnet3.autoresearch.executors.base import JobHandle, JobSpec


class LocalExecutor:
    """Run jobs via subprocess.Popen."""

    def __init__(self, max_jobs: int = 4):
        self.max_jobs = int(max_jobs)

    def submit(self, spec: JobSpec) -> JobHandle:
        spec.stdout.parent.mkdir(parents=True, exist_ok=True)
        spec.stderr.parent.mkdir(parents=True, exist_ok=True)
        stdout = open(spec.stdout, "a", encoding="utf-8")
        stderr = open(spec.stderr, "a", encoding="utf-8")
        process = subprocess.Popen(
            spec.command,
            cwd=str(spec.workdir),
            env={**os.environ, **spec.env},
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=True,
        )
        return JobHandle(
            backend="local",
            external_id=str(process.pid),
            metadata=spec.metadata,
        )

    def status(self, handle: JobHandle):
        marker_dir = handle.metadata.get("marker_dir")
        if marker_dir:
            marker_root = Path(marker_dir)
            if (marker_root / ".stage_success.json").exists():
                return "completed"
            if (marker_root / ".stage_failure.json").exists():
                return "failed"

        try:
            os.kill(int(handle.external_id), 0)
        except ProcessLookupError:
            return "unknown"
        except PermissionError:
            return "running"
        return "running"

    def cancel(self, handle: JobHandle) -> None:
        try:
            os.killpg(int(handle.external_id), signal.SIGTERM)
        except ProcessLookupError:
            return
