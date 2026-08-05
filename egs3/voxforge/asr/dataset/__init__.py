"""Dataset entry point for voxforge."""

from egs3.recipe_common.kaldi_dataset import KaldiDataBuilder
from egs3.recipe_common.kaldi_dataset import KaldiDataDataset


class Dataset(KaldiDataDataset):
    """Read the Kaldi data directory prepared by the egs2 recipe."""

    split_aliases = {'train':'tr_it','dev':'dt_it','test':'et_it'}


class DatasetBuilder(KaldiDataBuilder):
    """Validate the Kaldi data directory prepared by the egs2 recipe."""

    source_env_var = "VOXFORGE"
    split_aliases = Dataset.split_aliases


__all__ = ["Dataset", "DatasetBuilder"]
