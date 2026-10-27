"""Failures release owned shard locks and close successfully opened writers."""

from test.espnet3.parallel.test_base_runner_batch import DummyProvider

import pytest

from espnet3.parallel.base_runner import BaseRunner


class _FailureRunner(BaseRunner):
    handles = []
    fail_open = False

    @staticmethod
    def forward(idx, **env):
        raise ValueError("forward failure")

    @classmethod
    def open_writers(cls, shard_dir, **env):
        if cls.fail_open:
            raise ValueError("open failure")
        handle = (shard_dir / "output.txt").open("w")
        cls.handles.append(handle)
        return {"handle": handle}

    @staticmethod
    def close_writers(writers, state, **env):
        writers["handle"].close()


@pytest.mark.parametrize("fail_open", [False, True])
def test_failure_unlocks_unstarted_shards(tmp_path, fail_open):
    _FailureRunner.handles = []
    _FailureRunner.fail_open = fail_open
    runner = _FailureRunner(DummyProvider(), output_dir=tmp_path)
    shards = [{"shard_id": i, "items": [i]} for i in range(2)]
    pending = runner._filter_pending_shards(shards)
    with pytest.raises(ValueError):
        runner._run_local(pending)
    assert not list(tmp_path.rglob("lock"))
    assert not list(tmp_path.rglob("done"))
    assert all(handle.closed for handle in _FailureRunner.handles)


def test_lock_collision_rolls_back_only_owned_locks(tmp_path):
    runner = _FailureRunner(DummyProvider(), output_dir=tmp_path)
    directory = runner._resolve_shard_dir(str(tmp_path), "", 1)
    directory.mkdir(parents=True)
    lock = runner._get_lock_path(directory)
    lock.write_text("other")
    with pytest.raises(RuntimeError, match="locked"):
        runner._filter_pending_shards([{"shard_id": i} for i in range(2)])
    assert lock.read_text() == "other"
    assert len(list(tmp_path.rglob("lock"))) == 1
