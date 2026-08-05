"""Dataset entry point for vctk."""

from egs3.recipe_common.kaldi_dataset import KaldiDataBuilder
from egs3.recipe_common.kaldi_dataset import KaldiDataDataset


class Dataset(KaldiDataDataset):
    """Read the Kaldi data directory prepared by the egs2 recipe."""

    split_aliases = {'train':'tr_no_dev','dev':'dev','test':'test'}


class DatasetBuilder(KaldiDataBuilder):
    """Validate the Kaldi data directory prepared by the egs2 recipe."""

    source_env_var = "VCTK"
    split_aliases = Dataset.split_aliases


__all__ = ["Dataset", "DatasetBuilder"]
