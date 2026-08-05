"""Dataset entry point for WenetSpeech."""

from egs3.recipe_common.kaldi_dataset import KaldiDataBuilder
from egs3.recipe_common.kaldi_dataset import KaldiDataDataset


class Dataset(KaldiDataDataset):
    """Read the Kaldi data directory prepared by the egs2 recipe."""


class DatasetBuilder(KaldiDataBuilder):
    """Validate the Kaldi data directory prepared by the egs2 recipe."""

    source_env_var = "WENETSPEECH"


__all__ = ["Dataset", "DatasetBuilder"]
