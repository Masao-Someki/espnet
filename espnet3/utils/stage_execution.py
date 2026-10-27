"""Coordinate rank-zero stage side effects before and after DDP initialization."""

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch


def _write_json(path, value):
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _external_world():
    for rank_key, size_key in (
        ("RANK", "WORLD_SIZE"),
        ("SLURM_PROCID", "SLURM_NTASKS"),
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
    ):
        if rank_key in os.environ and int(os.environ.get(size_key, "1")) > 1:
            return int(os.environ[rank_key]), int(os.environ[size_key])
    return None


@contextmanager
def stage_execution(exp_dir, stage, rank, *, dry_run=False):
    """Execute non-training stages once, waiting for artifacts and sharing failures.

    Initialized groups use a broadcast. External launchers that have not yet
    entered Lightning use the shared experiment filesystem; a fresh arrival
    token and acknowledged generation prevent stale completion files from being
    reused. Lightning's own local child scripts start only after rank zero has
    completed prerequisite stages, so they skip those bodies without waiting.
    ``ESPNET3_STAGE_SYNC_TIMEOUT`` bounds filesystem waits (default: 24 hours).
    """
    if stage == "train" or dry_run:
        yield True
        return
    if torch.distributed.is_initialized():
        status = [None]
        if torch.distributed.get_rank() == 0:
            try:
                yield True
            except BaseException as exc:
                status[0] = f"{type(exc).__name__}: {exc}"
                torch.distributed.broadcast_object_list(status, src=0)
                raise
            else:
                torch.distributed.broadcast_object_list(status, src=0)
        else:
            torch.distributed.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise RuntimeError(f"Stage {stage!r} failed on rank zero: {status[0]}")
            yield False
        return

    external = _external_world()
    if external is None:
        yield rank == 0
        return
    rank, size = external
    run_id = [
        os.environ.get(key, "")
        for key in (
            "TORCHELASTIC_RUN_ID",
            "SLURM_JOB_ID",
            "SLURM_STEP_ID",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
    ]
    directory = (
        Path(exp_dir)
        / ".stage_sync"
        / hashlib.sha256(json.dumps(run_id).encode()).hexdigest()[:16]
        / stage
    )
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    _write_json(directory / f"arrival.{rank}.json", token)
    deadline = time.monotonic() + float(
        os.environ.get("ESPNET3_STAGE_SYNC_TIMEOUT", "86400")
    )
    generation = None
    arrivals = None
    while time.monotonic() < deadline:
        if rank == 0:
            current = [_read_json(directory / f"arrival.{i}.json") for i in range(size)]
            if all(current) and current != arrivals:
                arrivals = current
                generation = uuid.uuid4().hex
                _write_json(
                    directory / "generation.json",
                    {
                        "id": generation,
                        "arrivals": arrivals,
                    },
                )
            if generation and all(
                _read_json(directory / f"ack.{i}.json") == generation
                for i in range(1, size)
            ):
                break
        else:
            current = _read_json(directory / "generation.json")
            if (
                current
                and len(current["arrivals"]) == size
                and current["arrivals"][rank] == token
            ):
                if current["id"] != generation:
                    generation = current["id"]
                    _write_json(directory / f"ack.{rank}.json", generation)
                result = _read_json(directory / "result.json")
                if result and result["id"] == generation:
                    if result["error"] is not None:
                        raise RuntimeError(
                            f"Stage {stage!r} failed on rank zero: {result['error']}"
                        )
                    yield False
                    return
        time.sleep(0.02)
    else:
        raise TimeoutError(f"Timed out coordinating stage {stage!r} in {directory}")

    try:
        yield True
    except BaseException as exc:
        _write_json(
            directory / "result.json",
            {
                "id": generation,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    else:
        _write_json(directory / "result.json", {"id": generation, "error": None})
