"""F5-TTS system: the staged pipeline of an F5-TTS recipe.

On top of :class:`~espnet3.systems.base.system.BaseSystem` this adds the
two data-preparation stages an F5-TTS recipe runs between
``create_dataset`` and ``collect_stats``: ``remove_long_short`` and
``create_token_list``.
"""

import logging
from typing import ClassVar

from espnet3.components.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.f5tts.create_token_list import create_token_list
from espnet3.systems.f5tts.remove_long_short import remove_long_short

logger = logging.getLogger(__name__)


class F5TTSSystem(BaseSystem):
    """System for recipes that train :class:`espnet3.systems.f5tts.f5tts.F5TTS`.

    The stage order of an F5-TTS recipe is ``create_dataset ->
    remove_long_short -> create_token_list -> collect_stats -> train ->
    infer -> measure -> pack_model -> upload_model -> pack_demo ->
    upload_demo``. Every stage other than the two added here is inherited
    from ``BaseSystem`` unchanged: the model is instantiated directly from
    ``training_config.model._target_``, with ``task`` left unset.

    Examples:
        >>> [s.name for s in F5TTSSystem.stages[:3]]
        ['create_dataset', 'remove_long_short', 'create_token_list']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec("create_dataset", "training", "data_dir"),
        StageSpec("remove_long_short", "training", "remove_long_short.save_path"),
        StageSpec("create_token_list", "training", "create_token_list.save_path"),
        StageSpec("collect_stats", "training", "stats_dir"),
        StageSpec("train", "training", "exp_dir"),
        StageSpec("infer", "inference", "inference_dir"),
        StageSpec("measure", "metrics", "inference_dir"),
        StageSpec("pack_model", "publication"),
        StageSpec("upload_model", "publication"),
        StageSpec("pack_demo", "demo", "pack.out_dir"),
        StageSpec("upload_demo", "demo", "pack.out_dir"),
    )

    def remove_long_short(self, *args, **kwargs):
        """Filter the recipe's manifests by audio duration.

        Runs the ``remove_long_short`` stage on ``training_config``. See
        :func:`espnet3.systems.f5tts.remove_long_short.remove_long_short`
        for the ``training_config.remove_long_short`` fields and the files
        written.

        Raises:
            TypeError: If any positional or keyword argument is passed.
            RuntimeError: If required configuration is missing or a manifest
                file is not found.

        Examples:
            >>> system = F5TTSSystem(training_config=training_config)
            >>> system.remove_long_short()

        Note:
            Run it after ``create_dataset``, which writes the manifests it
            reads, and before ``create_token_list``, which reads the filtered
            training manifest.
        """
        self._reject_stage_args("remove_long_short", args, kwargs)
        logger.info("F5TTSSystem.remove_long_short(): starting duration filtering")
        return remove_long_short(self.training_config)

    def create_token_list(self, *args, **kwargs):
        """Build the token list from the training manifest.

        Runs the ``create_token_list`` stage on ``training_config``. See
        :func:`espnet3.systems.f5tts.create_token_list.create_token_list`
        for the ``training_config.create_token_list`` fields and the file
        written.

        Raises:
            TypeError: If any positional or keyword argument is passed.
            RuntimeError: If required configuration is missing or the
                manifest file is not found.

        Examples:
            >>> system = F5TTSSystem(training_config=training_config)
            >>> system.create_token_list()

        Note:
            The model and the dataset preprocessor both read the token list
            this stage writes, so it has to run before ``collect_stats``.
        """
        self._reject_stage_args("create_token_list", args, kwargs)
        logger.info("F5TTSSystem.create_token_list(): starting token list creation")
        return create_token_list(self.training_config)
