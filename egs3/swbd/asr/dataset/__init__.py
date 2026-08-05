"""Dataset entry point for swbd."""

from egs3.recipe_common.kaldi_dataset import KaldiDataBuilder
from egs3.recipe_common.kaldi_dataset import KaldiDataDataset


class Dataset(KaldiDataDataset):
    """Read the Kaldi data directory prepared by the egs2 recipe."""

    split_aliases = {'train':'train_nodup','dev':'train_dev','test':'eval2000'}


class DatasetBuilder(KaldiDataBuilder):
    """Validate the Kaldi data directory prepared by the egs2 recipe."""

    source_env_var = "SWBD"
    split_aliases = Dataset.split_aliases


__all__ = ["Dataset", "DatasetBuilder"]
