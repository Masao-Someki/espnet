"""Run a trial's `trial.commands` sequentially via subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from espnet3.autoresearch.config import render, render_list


@dataclass
class RunResult:
    """Outcome of running one trial's `trial.commands`."""

    status: str  # "success" | "failed" | "timeout"
    returncode: Optional[int] = None
    failed_command_index: Optional[int] = None
    message: str = ""


def run_trial(
    *,
    commands: Sequence[Sequence[str]],
    workdir: str,
    env: Mapping[str, str],
    timeout_sec: int,
    placeholders: Mapping[str, Any],
    trial_dir: Path,
) -> RunResult:
    """Render and run each of `commands` in order, logging to `trial_dir`.

    Stops at the first command that exits non-zero, or when the remaining
    share of the combined `timeout_sec` budget (shared across all commands)
    runs out before a command finishes.

    Args:
        commands: `trial.commands`: a list of argv lists, each argument a
            `{placeholder}` template string.
        workdir: `{placeholder}` template for the subprocess working directory.
        env: Extra environment variables (template values), merged over a
            copy of this process's environment.
        timeout_sec: Combined wall-clock budget for all commands together.
        placeholders: Values available to `{name}` templates (see
            `config.KNOWN_PLACEHOLDERS`).
        trial_dir: Trial directory; `commands.json` (the rendered argv for
            every command) and one `cmd_NN.log` (combined stdout+stderr) per
            command are written here.

    Returns:
        RunResult: `status="success"` if every command exited 0;
        `status="failed"` with `failed_command_index` set to the first
        non-zero command; `status="timeout"` if the combined budget ran out
        before all commands finished.

    Raises:
        ValueError: If a command, `workdir`, or an `env` value contains a
            `{name}` placeholder not present in `placeholders`.
    """
    rendered_workdir = render(workdir, placeholders)
    rendered_env = {k: render(v, placeholders) for k, v in env.items()}
    full_env = {**os.environ, **rendered_env}

    rendered_commands = [render_list(cmd, placeholders) for cmd in commands]
    (trial_dir / "commands.json").write_text(
        json.dumps(rendered_commands, indent=2), encoding="utf-8"
    )

    deadline = time.monotonic() + timeout_sec
    for index, argv in enumerate(rendered_commands):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return RunResult(
                status="timeout",
                failed_command_index=index,
                message=f"Timed out before running command {index}: {argv}",
            )
        log_path = trial_dir / f"cmd_{index:02d}.log"
        try:
            with open(log_path, "wb") as log_file:
                proc = subprocess.run(
                    argv,
                    cwd=rendered_workdir,
                    env=full_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                )
        except subprocess.TimeoutExpired:
            return RunResult(
                status="timeout",
                failed_command_index=index,
                message=f"Command {index} timed out: {argv}",
            )
        if proc.returncode != 0:
            return RunResult(
                status="failed",
                returncode=proc.returncode,
                failed_command_index=index,
                message=f"Command {index} exited {proc.returncode}: {argv}",
            )
    return RunResult(status="success")
