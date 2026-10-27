"""Rank-exclusive shards use local shape IDs and retain every owned batch."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from espnet2.train.collate_fn import CommonCollateFn
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.components.data.collect_stats import collect_stats
from espnet3.components.data.data_organizer import DataOrganizer
from espnet3.components.data.dataloader import DataLoaderBuilder
from espnet3.components.data.dataset import ShardedDataset
from espnet3.contract import Field


@pytest.fixture(autouse=True)
def initialized_topology(monkeypatch):
    """Simulate collective calls while checking every rank's actual samples."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)


class _Partition(ShardedDataset, BaseDataset):
    fields = (Field("speech", "audio"),)
    total_shards = 2
    dist_world_size = 2

    def __init__(self, indices=None):
        self.indices = list(range(6)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        value = self.indices[index] + 1
        return {"speech": np.full((value, 1), value, dtype=np.float32)}

    def shard(self, index):
        return _Partition([[5, 0, 3], [1, 4, 2]][index])


Dataset = _Partition


class _Model(torch.nn.Module):
    def collect_feats(self, speech, speech_lengths):
        return {"feats": speech, "feats_lengths": speech_lengths}


def _dataset_config():
    entry = {"name": "data", "data_src": __name__}
    return OmegaConf.create(
        {
            "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
            "_recursive_": False,
            "train": [entry],
            "valid": [entry],
        }
    )


@pytest.mark.parametrize("iterator", [False, True])
def test_sharded_sample_coverage(tmp_path, monkeypatch, iterator):
    ds_config = _dataset_config()
    collect_stats(
        {"_target_": f"{__name__}._Model"},
        ds_config,
        OmegaConf.create({}),
        "train",
        tmp_path,
    )
    statistics = np.load(tmp_path / "train/feats_stats.npz")
    assert statistics["count"] == 21
    np.testing.assert_array_equal(statistics["sum"], [91])
    collate = CommonCollateFn()
    organizer = DataOrganizer(train=ds_config.train, valid=ds_config.valid)
    organizer.train.use_espnet_collator = True
    config = OmegaConf.create(
        {
            "dataloader": {
                "train": {
                    "iter_factory": None,
                    "batch_size": 1,
                    "shuffle": False,
                }
            }
        }
    )
    if iterator:
        config.dataloader.train.iter_factory = {
            "_target_": (
                "espnet2.iterators.sequence_iter_factory." "SequenceIterFactory"
            ),
            "shuffle": False,
            "num_workers": 0,
            "collate_fn": {"_target_": "espnet2.train.collate_fn.CommonCollateFn"},
            "batches": {
                "type": "unsorted",
                "batch_size": 1,
                "batch_bins": 0,
                "shape_files": [str(tmp_path / "shard.{shard_idx}/train/speech_shape")],
            },
        }
    all_values = []
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    for rank in [0, 1]:
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        loader = DataLoaderBuilder(organizer.train, config, collate, 2, 0).build(
            "train"
        )
        values = [int(batch[1]["speech"][0, 0, 0]) for batch in loader]
        assert len(values) == 3
        all_values.extend(values)
    assert sorted(all_values) == list(range(1, 7))


def test_sharded_iterator_rejects_global_shape_ids(tmp_path, monkeypatch):
    config = OmegaConf.create(
        {
            "dataloader": {
                "train": {
                    "iter_factory": {
                        "_target_": (
                            "espnet2.iterators.sequence_iter_factory."
                            "SequenceIterFactory"
                        ),
                        "batches": {"shape_files": [str(tmp_path / "global_shape")]},
                    },
                }
            }
        }
    )
    ds_config = _dataset_config()
    dataset = DataOrganizer(train=ds_config.train, valid=ds_config.valid).train
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    with pytest.raises(RuntimeError, match="shard-local shape_files"):
        DataLoaderBuilder(dataset, config, None, 2, 0).build("train")


def test_rank_sharded_standard_loader_rejects_unequal_batch_counts(monkeypatch):
    config = OmegaConf.create(
        {
            "dataloader": {
                "train": {
                    "iter_factory": None,
                    "batch_size": 1,
                    "shuffle": False,
                }
            }
        }
    )
    ds_config = _dataset_config()
    dataset = DataOrganizer(train=ds_config.train, valid=ds_config.valid).train
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

    def global_counts(counts, op):
        counts[0], counts[1] = 2, -3

    monkeypatch.setattr(torch.distributed, "all_reduce", global_counts)
    with pytest.raises(RuntimeError, match="equal batch counts"):
        DataLoaderBuilder(dataset, config, None, 2, 0).build("train")
