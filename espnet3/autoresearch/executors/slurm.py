"""Slurm executor."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from espnet3.autoresearch.core.errors import JobSubmissionError
from espnet3.autoresearch.core.serialization import atomic_write_text
from espnet3.autoresearch.executors.base import JobHandle, JobSpec
from espnet3.autoresearch.executors.script import build_shell_script
from espnet3.autoresearch.executors.status import map_slurm_state

_JOB_ID_RE = re.compile(r"Submitted batch job (?P<job_id>\d+)")


@dataclass
class SlurmConfig:
    """Static Slurm executor config."""

    sbatch: str = "sbatch"
    squeue: str = "squeue"
    sacct: str = "sacct"
    launcher: str = ""

    @classmethod
    def from_config(cls, config) -> "SlurmConfig":
        return cls(
            sbatch=str(getattr(config, "sbatch", "sbatch")),
            squeue=str(getattr(config, "squeue", "squeue")),
            sacct=str(getattr(config, "sacct", "sacct")),
            launcher=str(getattr(config, "launcher", "") or ""),
        )


class SlurmExecutor:
    """Submit node jobs to Slurm."""

    def __init__(self, config: SlurmConfig):
        self.config = config

    def _generate_script(self, spec: JobSpec) -> str:
        resources = spec.resources
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={spec.name}",
            f"#SBATCH --output={spec.stdout}",
            f"#SBATCH --error={spec.stderr}",
            f"#SBATCH --chdir={spec.workdir}",
        ]
        if resources.time:
            lines.append(f"#SBATCH --time={resources.time}")
        if resources.cpus:
            lines.append(f"#SBATCH --cpus-per-task={resources.cpus}")
        if resources.nodes:
            lines.append(f"#SBATCH --nodes={resources.nodes}")
        if resources.gpus:
            lines.append(f"#SBATCH --gres=gpu:{resources.gpus}")
        if resources.mem:
            lines.append(f"#SBATCH --mem={resources.mem}")
        if resources.partition:
            lines.append(f"#SBATCH --partition={resources.partition}")
        if resources.account:
            lines.append(f"#SBATCH --account={resources.account}")
        if resources.qos:
            lines.append(f"#SBATCH --qos={resources.qos}")
        if resources.constraint:
            lines.append(f"#SBATCH --constraint={resources.constraint}")
        if resources.reservation:
            lines.append(f"#SBATCH --reservation={resources.reservation}")
        if resources.nodelist:
            lines.append(f"#SBATCH --nodelist={resources.nodelist}")
        for key, value in resources.extra.items():
            lines.append(f"#SBATCH --{key}={value}")
        lines.append("")
        lines.append(build_shell_script(spec, launcher=self.config.launcher))
        return "\n".join(lines)

    def _write_submit_logs(self, spec: JobSpec, stdout_text: str, stderr_text: str) -> None:
        stdout_path = spec.metadata.get("submit_stdout_path")
        stderr_path = spec.metadata.get("submit_stderr_path")
        if stdout_path:
            atomic_write_text(Path(stdout_path), stdout_text)
        if stderr_path:
            atomic_write_text(Path(stderr_path), stderr_text)

    def submit(self, spec: JobSpec) -> JobHandle:
        spec.stdout.parent.mkdir(parents=True, exist_ok=True)
        spec.stderr.parent.mkdir(parents=True, exist_ok=True)
        script_path = Path(spec.metadata["script_path"])
        atomic_write_text(script_path, self._generate_script(spec))
        script_path.chmod(0o755)
        proc = subprocess.run(
            [self.config.sbatch, str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        self._write_submit_logs(spec, proc.stdout, proc.stderr)
        if proc.returncode != 0:
            raise JobSubmissionError(proc.stderr.strip() or proc.stdout.strip())
        match = _JOB_ID_RE.search(proc.stdout)
        if match is None:
            raise JobSubmissionError(
                f"Failed to parse sbatch output for job id: {proc.stdout!r}"
            )
        return JobHandle(
            backend="slurm",
            external_id=match.group("job_id"),
            metadata={
                **spec.metadata,
                "submit_stdout": proc.stdout,
                "submit_stderr": proc.stderr,
            },
        )

    def status(self, handle: JobHandle):
        job_id = handle.external_id
        squeue = subprocess.run(
            [self.config.squeue, "-h", "-j", str(job_id), "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
        )
        state = squeue.stdout.strip()
        if state:
            return map_slurm_state(state)
        sacct = subprocess.run(
            [self.config.sacct, "-n", "-X", "-j", str(job_id), "-o", "State"],
            check=False,
            capture_output=True,
            text=True,
        )
        state = sacct.stdout.strip().splitlines()
        if state:
            return map_slurm_state(state[0])
        return "unknown"

    def cancel(self, handle: JobHandle) -> None:
        subprocess.run(["scancel", str(handle.external_id)], check=False)
