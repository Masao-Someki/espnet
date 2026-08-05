"""Dataset entry point for fisher_callhome_spanish."""

from egs3.recipe_common.kaldi_dataset import KaldiDataBuilder
from egs3.recipe_common.kaldi_dataset import KaldiDataDataset


class Dataset(KaldiDataDataset):
    """Read the Kaldi data directory prepared by the egs2 recipe."""

    split_aliases = {}


class DatasetBuilder(KaldiDataBuilder):
    """Validate the Kaldi data directory prepared by the egs2 recipe."""

    source_env_var = "FISHER_CALLHOME_SPANISH"
    split_aliases = Dataset.split_aliases


__all__ = ["Dataset", "DatasetBuilder"]
