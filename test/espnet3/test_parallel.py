import sys
import types
import time
import pytest
from contextlib import nullcontext
from omegaconf import OmegaConf

from espnet3.parallel import (  # ← モジュールのインポートパスに合わせて直してね
    set_parallel,
    get_parallel_config,
    make_client,
    get_client,
    parallel_map,
    parallel_for,
    wrap_func_with_worker_env,
    make_local_gpu_cluster,
)

import multiprocessing as mp
mp.set_start_method("fork", force=True)

# --------- Fixtures ---------

@pytest.fixture
def local_cfg():
    # LocalCluster だけを使う設定
    cfg = OmegaConf.create(
        {
            "env": "local",
            "n_workers": 2,
            # 安全側に寄せるオプション（CI で安定化）
            "options": {
                "threads_per_worker": 1,
                "processes": True,
                # "silence_logs": "error",
                # "dashboard_address": None,
            },
        }
    )
    return cfg


@pytest.fixture
def set_global_parallel(local_cfg):
    # set_parallel / get_parallel_config の正常系
    set_parallel(local_cfg)
    yield
    # 後片付け（必要なら parallel 側に reset 関数があれば呼ぶ）
    set_parallel(local_cfg)  # ダミーで再設定しても実害なし


# --------- 正常系 ---------

def test_set_and_get_parallel_config(local_cfg):
    set_parallel(local_cfg)
    got = get_parallel_config()
    assert got.env == "local"
    assert got.n_workers == 2
    assert "threads_per_worker" in got.options


def test_make_client_local(local_cfg):
    client = make_client(local_cfg)
    try:
        futs = client.map(lambda x: x * x, range(5))
        out = client.gather(futs)
        assert out == [0, 1, 4, 9, 16]
    finally:
        client.close()
