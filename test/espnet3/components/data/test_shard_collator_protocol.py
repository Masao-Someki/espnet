"""Selecting a dataset shard preserves the configured collator protocol."""

import pytest

from espnet3.components.data.dataset import CombinedDataset, ShardedDataset


class _Shards(ShardedDataset):
    total_shards = 2
    dist_world_size = 1

    def __init__(self, index=0):
        self.index = index

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"value": self.index}

    def shard(self, index):
        return _Shards(index)


def _identity(sample):
    return sample


@pytest.mark.parametrize("collator", [False, True])
def test_shard_preserves_collator_protocol(collator):
    """Both tuple and dictionary return formats survive repeated selection."""
    dataset = CombinedDataset([_Shards()], [(_identity, _identity)], False)
    dataset.use_espnet_collator = collator
    for index in [1, 0]:
        dataset = dataset.shard(index)
        assert dataset.use_espnet_collator is collator
        expected = {"value": index}
        assert dataset[0] == (("0", expected) if collator else expected)
