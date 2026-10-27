"""Real two-process stage ownership, artifact ordering and failure propagation."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from espnet3.utils.stage_execution import stage_execution


def _stage_worker(rank, root, initialized, fail, queue):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", ESPNET3_STAGE_SYNC_TIMEOUT="15")
    if initialized:
        torch.distributed.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
            init_method=f"file://{root}/rendezvous",
            timeout=timedelta(seconds=20),
        )
    try:
        try:
            with stage_execution(root, "prepare", rank) as execute:
                if execute:
                    with (Path(root) / "side_effect.txt").open("a") as handle:
                        handle.write("once\n")
                    if fail:
                        raise ValueError("deliberate failure")
            queue.put((rank, "ready", (Path(root) / "side_effect.txt").read_text()))
            with stage_execution(root, "train", rank) as execute:
                assert execute
        except (RuntimeError, ValueError) as exc:
            queue.put((rank, "error", str(exc)))
    finally:
        if initialized:
            torch.distributed.destroy_process_group()


@pytest.mark.execution_timeout(60)
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_two_rank_stage_execution(tmp_path, initialized, fail):
    for rerun in range(2):
        # Reuse the same sync directory: old completions must never satisfy a new run.
        queue = mp.get_context("fork").Queue()
        mp.start_processes(
            _stage_worker,
            args=(str(tmp_path), initialized, fail, queue),
            nprocs=2,
            start_method="fork",
            join=True,
        )
        messages = [queue.get(timeout=5) for _ in range(2)]
        queue.close()
        assert {message[0] for message in messages} == {0, 1}
        assert all(message[1] == ("error" if fail else "ready") for message in messages)
        if fail:
            assert all("deliberate failure" in message[2] for message in messages)
        assert (tmp_path / "side_effect.txt").read_text() == "once\n" * (rerun + 1)


def test_lightning_relaunched_child_skips_setup(monkeypatch, tmp_path):
    for key in ["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with stage_execution(tmp_path, "collect_stats", 1) as execute:
        assert not execute
    with stage_execution(tmp_path, "train", 1) as execute:
        assert execute
