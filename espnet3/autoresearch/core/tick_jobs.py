"""Helpers for long-running tick and trial-monitor jobs."""

from __future__ import annotations

import shlex
from pathlib import Path


def parse_slurm_time_to_seconds(value: str | None) -> int | None:
    """Parse a Slurm time string into seconds."""
    if value in {None, ""}:
        return None
    text = str(value).strip()
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 1:
        hours = 0
        minutes = parts[0]
        seconds = 0
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported Slurm time format: {value}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def compute_tick_handoff_after_seconds(walltime_seconds: int | None) -> int | None:
    """Return when a long-running tick job should hand off to its successor."""
    if walltime_seconds is None or walltime_seconds <= 0:
        return None
    cushion = min(1800, max(60, walltime_seconds // 4))
    return max(1, walltime_seconds - cushion)


def build_tick_loop_script(
    *,
    study_dir: Path,
    recipe_dir: Path,
    interval: int,
    backend: str,
    slurm_time: str | None,
    slurm_partition: str | None,
    slurm_account: str | None,
    slurm_mem: str | None,
    slurm_qos: str | None,
    slurm_constraint: str | None,
    slurm_reservation: str | None,
    slurm_nodelist: str | None,
    trial_id: str | None,
) -> str:
    """Build a bash script for a self-renewing tick job."""
    quoted_study_dir = shlex.quote(str(study_dir))
    quoted_recipe_dir = shlex.quote(str(recipe_dir))
    tick_cmd = [
        "pixi",
        "run",
        "python",
        "-m",
        "espnet3.autoresearch.cli.main",
        "tick",
        "--study-dir",
        quoted_study_dir,
        "--recipe-dir",
        quoted_recipe_dir,
    ]
    if trial_id:
        tick_cmd.extend(["--trial-id", shlex.quote(str(trial_id))])

    resubmit_cmd = [
        "pixi",
        "run",
        "python",
        "-m",
        "espnet3.autoresearch.cli.main",
        "schedule-next-tick",
        "--study-dir",
        quoted_study_dir,
        "--recipe-dir",
        quoted_recipe_dir,
        "--interval",
        "0",
        "--backend",
        backend,
    ]
    if trial_id:
        resubmit_cmd.extend(["--trial-id", shlex.quote(str(trial_id))])
    if slurm_time:
        resubmit_cmd.extend(["--slurm-time", shlex.quote(str(slurm_time))])
    if slurm_partition:
        resubmit_cmd.extend(["--slurm-partition", shlex.quote(str(slurm_partition))])
    if slurm_account:
        resubmit_cmd.extend(["--slurm-account", shlex.quote(str(slurm_account))])
    if slurm_mem:
        resubmit_cmd.extend(["--slurm-mem", shlex.quote(str(slurm_mem))])
    if slurm_qos:
        resubmit_cmd.extend(["--slurm-qos", shlex.quote(str(slurm_qos))])
    if slurm_constraint:
        resubmit_cmd.extend(["--slurm-constraint", shlex.quote(str(slurm_constraint))])
    if slurm_reservation:
        resubmit_cmd.extend(
            ["--slurm-reservation", shlex.quote(str(slurm_reservation))]
        )
    if slurm_nodelist:
        resubmit_cmd.extend(
            ["--slurm-nodelist", shlex.quote(str(slurm_nodelist))]
        )

    handoff_after = compute_tick_handoff_after_seconds(parse_slurm_time_to_seconds(slurm_time))
    lines = [
        'finish_tick_job() {',
        '  if printenv ESPNET_AR_TICK_JOB_ID >/dev/null 2>&1; then',
        "    pixi run python -m espnet3.autoresearch.cli.main finish-tick-job "
        f"--study-dir {quoted_study_dir} "
        '--job-id "$ESPNET_AR_TICK_JOB_ID" || true',
        "  fi",
        "}",
        "trap finish_tick_job EXIT",
    ]
    if handoff_after is not None:
        lines.extend(
            [
                'start_ts="$(date +%s)"',
                f"handoff_after_sec={int(handoff_after)}",
            ]
        )
    lines.extend(
        [
            "while true; do",
            f"  {' '.join(tick_cmd)}",
        ]
    )
    if handoff_after is not None:
        lines.extend(
            [
                '  now_ts="$(date +%s)"',
                '  elapsed_sec="$((now_ts - start_ts))"',
                '  if [ "$elapsed_sec" -ge "$handoff_after_sec" ]; then',
                f"    if {' '.join(resubmit_cmd)}; then",
                "      exit 0",
                "    fi",
                "  fi",
            ]
        )
    if interval > 0:
        lines.append(f"  sleep {int(interval)}")
    lines.append("done")
    return "\n".join(lines)
