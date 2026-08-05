"""Dataset entry point for fleurs."""

from egs3.recipe_common.kaldi_dataset import KaldiDataBuilder
from egs3.recipe_common.kaldi_dataset import KaldiDataDataset


class Dataset(KaldiDataDataset):
    """Read the Kaldi data directory prepared by the egs2 recipe."""

    split_aliases = {'train':'train_af_za','dev':'dev_af_za','test':'test_af_za'}


class DatasetBuilder(KaldiDataBuilder):
    """Validate the Kaldi data directory prepared by the egs2 recipe."""

    source_env_var = "FLEURS"
    split_aliases = Dataset.split_aliases


__all__ = ["Dataset", "DatasetBuilder"]
