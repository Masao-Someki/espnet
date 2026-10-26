"""Statistics use the dataset's collator protocol independently of preprocessing."""

import numpy as np
import pytest

from espnet2.train.collate_fn import CommonCollateFn
from espnet3.components.data.collect_stats import collect_stats_batch
from espnet3.components.data.dataset import CombinedDataset


class _Features:
    def collect_feats(self, speech, speech_lengths):
        return {"feats": speech, "feats_lengths": speech_lengths}


def _identity(sample):
    return sample


def _preprocess(uid, sample):
    assert uid in {"0", "1"}
    return sample


@pytest.mark.parametrize("preprocessor", [False, True])
@pytest.mark.parametrize("collator", [False, True])
def test_stats_tuple_protocol(preprocessor, collator):
    """All four independent flag combinations preserve IDs and feature sums."""
    samples = [
        {"speech": np.full((3, 1), 1, dtype=np.float32)},
        {"speech": np.full((5, 1), 2, dtype=np.float32)},
    ]
    dataset = CombinedDataset(
        [samples],
        [(_identity, _preprocess if preprocessor else _identity)],
        preprocessor,
    )
    dataset.use_espnet_collator = collator
    stats, shapes = collect_stats_batch(
        [0, 1], model=_Features(), dataset=dataset, collate_fn=CommonCollateFn()
    )
    assert shapes["speech"] == {"0": "3,1", "1": "5,1"}
    assert stats["feats"]["count"] == 8
    np.testing.assert_array_equal(stats["feats"]["sum"], [13])
    np.testing.assert_array_equal(stats["feats"]["sq"], [23])
