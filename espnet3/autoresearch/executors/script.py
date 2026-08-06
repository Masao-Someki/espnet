"""Shell script generation for backend jobs."""

from __future__ import annotations

import shlex

from espnet3.autoresearch.executors.base import JobSpec


def build_shell_script(spec: JobSpec, launcher: str = "") -> str:
    """Build a portable shell script for a job.

    ``launcher`` is a template string (e.g. ``"srun"`` or
    ``"torchrun --nnodes={nodes} --nproc_per_node={gpus}"``) prepended to the
    ``pixi run`` invocation. It supports ``{gpus}``, ``{cpus}``, ``{nodes}``,
    and ``{ntasks}`` (``{ntasks}`` defaults to ``max(gpus, 1)``, i.e. tasks
    per node) placeholders filled in from ``spec.resources``.
    """
    lines = ["#!/usr/bin/env bash", ""]
    if not launcher:
        # No external launcher (e.g. `srun`) creates the training processes,
        # so unset the SLURM task env vars sbatch still exports. Otherwise
        # PyTorch Lightning auto-detects SLURMEnvironment and refuses to
        # spawn its own multi-GPU subprocesses, expecting an external
        # launcher to have created them.
        lines.append(
            "unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_NPROCS "
            "SLURM_PROCID SLURM_LOCALID SLURM_NODEID"
        )
        lines.append("")
    for key, value in sorted(spec.env.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append(f"cd {shlex.quote(str(spec.workdir))}")
    launcher_tokens: list[str] = []
    if launcher:
        gpus = spec.resources.gpus
        launcher_expanded = launcher.format(
            gpus=gpus,
            cpus=spec.resources.cpus,
            nodes=spec.resources.nodes,
            ntasks=max(gpus, 1),
        )
        launcher_tokens = shlex.split(launcher_expanded)
    cmd = launcher_tokens + ["pixi", "run"] + [str(part) for part in spec.command]
    lines.append(" ".join(shlex.quote(str(part)) for part in cmd))
    lines.append("")
    return "\n".join(lines)
