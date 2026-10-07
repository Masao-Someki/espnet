"""Base system class and stage entrypoints for ESPnet3."""

import copy
import logging
import time
from pathlib import Path
from typing import ClassVar

from omegaconf import DictConfig, OmegaConf

from espnet3.components.contract.stages import StageSpec, check_stage_contract
from espnet3.components.data.dataset_module import (
    load_dataset_module,
    parse_dataset_reference_config,
)
from espnet3.publication.demo.packing import pack_demo as _pack_demo
from espnet3.publication.demo.packing import upload_demo as _upload_demo
from espnet3.systems.base.inference import infer
from espnet3.systems.base.metric import measure
from espnet3.systems.base.training import collect_stats, train
from espnet3.utils.publication_utils import pack_model as _pack_model
from espnet3.utils.publication_utils import upload_model as _upload_model

logger = logging.getLogger(__name__)


class BaseSystem:
    """Base class for all ESPnet3 systems.

    A system declares its stages as the class attribute ``stages``: a
    tuple of :class:`~espnet3.components.contract.stages.StageSpec`, in
    the order "all" runs them. Every public method a system defines
    (class-defined, not starting with ``_``) must be one of these stage
    names - a system has no other public surface. See
    ``espnet3.components.contract.stages.check_stage_contract`` for the
    checks run when a subclass is defined, and
    ``espnet3.components.contract.stages.stage_log_dir`` for how a
    stage's log directory is resolved.

    Class Attributes:
        DATASET_BUILDER_CLASS_NAME: Name of the builder class expected in each
            dataset module (default ``"DatasetBuilder"``).
        DATASET_CLASS_NAME: Name of the dataset class expected in each dataset
            module (default ``"Dataset"``). Used by subclasses that instantiate
            datasets directly.
        stages: The declared stage order (see above).

    All behavior is config-driven.

    Args:
        training_config (DictConfig | None): Training configuration.
        inference_config (DictConfig | None): Inference configuration.
        metrics_config (DictConfig | None): Measurement configuration.
        publication_config (DictConfig | None): Publication configuration.
        demo_config (DictConfig | None): Demo configuration.

    Examples:
        >>> system = BaseSystem()
        >>> [s.name for s in system.stages[:3]]
        ['create_dataset', 'collect_stats', 'train']
    """

    DATASET_BUILDER_CLASS_NAME = "DatasetBuilder"
    DATASET_CLASS_NAME = "Dataset"

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec("create_dataset", "training", "data_dir"),
        StageSpec("collect_stats", "training", "stats_dir"),
        StageSpec("train", "training", "exp_dir"),
        StageSpec("infer", "inference", "inference_dir"),
        StageSpec("measure", "metrics", "inference_dir"),
        StageSpec("pack_model", "publication"),
        StageSpec("upload_model", "publication"),
        StageSpec("pack_demo", "demo", "pack.out_dir"),
        StageSpec("upload_demo", "demo", "pack.out_dir"),
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """Check the subclass's declared ``stages`` against its own methods."""
        super().__init_subclass__(**kwargs)
        check_stage_contract(cls)

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        publication_config: DictConfig | None = None,
        demo_config: DictConfig | None = None,
    ) -> None:
        """Initialize the system with optional stage configs.

        Args:
            training_config: Training configuration for data preparation,
                statistics collection, and model training.
            inference_config: Inference configuration used by the ``infer``
                stage.
            metrics_config: Measurement configuration used by the ``measure``
                stage.
            publication_config: Publication configuration for ``pack_model``
                and ``upload_model`` stages.
            demo_config: Demo configuration for the ``demo`` stage.

        Notes:
            Each given config is marked readonly here and kept as-is for
            the system's lifetime; stages never see these objects directly.
            ``_stage_config`` hands each stage a writable deep copy instead,
            so one stage's in-place edits (or pops) never leak into another.
        """
        for config in (
            training_config,
            inference_config,
            metrics_config,
            publication_config,
            demo_config,
        ):
            if config is not None:
                OmegaConf.set_readonly(config, True)

        self.training_config = training_config
        self.inference_config = inference_config
        self.metrics_config = metrics_config
        self.publication_config = publication_config
        self.demo_config = demo_config

        if training_config is not None:
            self.exp_dir = Path(training_config.exp_dir)
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.exp_dir = None

        self._default_log_dir = self.exp_dir or (Path.cwd() / "logs")

        logger.info(
            "Initialized %s with training_config=%s inference_config=%s "
            "metrics_config=%s publication_config=%s demo_config=%s exp_dir=%s",
            self.__class__.__name__,
            training_config is not None,
            inference_config is not None,
            metrics_config is not None,
            publication_config is not None,
            demo_config is not None,
            self.exp_dir,
        )

    @staticmethod
    def _reject_stage_args(stage: str, args, kwargs) -> None:
        """Reject unexpected positional/keyword arguments for stages."""
        if args or kwargs:
            raise TypeError(
                f"Stage '{stage}' does not accept arguments. "
                "Put all settings in the YAML config."
            )

    @staticmethod
    def _get_required_config(config, key: str, error_message: str):
        """Return ``config[key]``, raising ``RuntimeError`` when missing.

        Args:
            config: Dict-like config (e.g. ``DictConfig``) to read from.
                ``None`` is treated the same as a missing key.
            key: Field name to extract.
            error_message: Message for the ``RuntimeError`` raised when the
                field is absent or ``None``.

        Returns:
            The value stored under ``key``.

        Raises:
            RuntimeError: If ``config`` is ``None`` or ``config[key]`` is
                missing or ``None``.
        """
        value = config.get(key, None) if config is not None else None
        if value is None:
            raise RuntimeError(error_message)
        return value

    @staticmethod
    def _stage_config(config: DictConfig | None) -> DictConfig | None:
        """Return a writable deep copy of a stage config, or `None`.

        The system's own ``*_config`` attributes stay readonly for the
        system's whole lifetime (see `__init__`); each stage call gets its
        own deep copy instead, so a stage that pops or overwrites a key
        (e.g. `collect_stats` dropping `model.normalize`, `trainer.py`
        setting `reload_dataloaders_every_n_epochs`) never affects the
        system's stored config or any other stage's copy.

        Args:
            config: A stage's config role (e.g. `self.training_config`), or
                `None` when that role was not given.

        Returns:
            A writable deep copy of `config`, or `None` if `config` is
            `None`.

        Examples:
            >>> from omegaconf import OmegaConf
            >>> original = OmegaConf.create({"a": 1})
            >>> OmegaConf.set_readonly(original, True)
            >>> copy_ = BaseSystem._stage_config(original)
            >>> copy_.a = 2
            >>> original.a
            1
        """
        if config is None:
            return None
        copy_ = copy.deepcopy(config)
        OmegaConf.set_readonly(copy_, False)
        return copy_

    # ---------------------------------------------------------
    # Stage stubs (override in subclasses if needed)
    # ---------------------------------------------------------
    def create_dataset(self, *args, **kwargs):
        """Create datasets from dataset references."""
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info(
            "%s.create_dataset(): starting dataset creation process",
            self.__class__.__name__,
        )
        start = time.perf_counter()
        dataset_config = getattr(self.training_config, "dataset", None)
        recipe_dir = getattr(self.training_config, "recipe_dir", None)
        create_dataset_config = getattr(
            self.training_config, "create_dataset", OmegaConf.create({})
        )
        default_builder_kwargs = dict(create_dataset_config)

        prepared_any = False

        if dataset_config is None:
            raise RuntimeError(
                "training_config.dataset must be set for create_dataset stage."
            )

        prepared_refs: set[str] = set()
        for split_name in ("train", "valid", "test"):
            entries = getattr(dataset_config, split_name, None)
            if entries is None:
                continue
            for entry in entries:
                plain = dict(entry)
                data_src, _ = parse_dataset_reference_config(plain)
                if data_src in prepared_refs:
                    continue
                prepared_refs.add(data_src)

                builder_kwargs = dict(default_builder_kwargs)

                module = load_dataset_module(data_src=data_src, recipe_dir=recipe_dir)
                builder = getattr(module, self.DATASET_BUILDER_CLASS_NAME)()
                logger.info("Ensuring dataset is prepared: %s", data_src or "local")

                # Ensure raw source exists first, then build task-ready artifacts.
                if not builder.is_source_prepared(**builder_kwargs):
                    builder.prepare_source(**builder_kwargs)

                if not builder.is_built(**builder_kwargs):
                    builder.build(**builder_kwargs)
                prepared_any = True

        if not prepared_any:
            raise RuntimeError(
                "training_config.dataset must include at least one entry in "
                "dataset.train / dataset.valid / dataset.test."
            )

        logger.info(
            "Dataset creation completed in %.2fs",
            time.perf_counter() - start,
        )
        return None

    def collect_stats(self, *args, **kwargs):
        """Collect statistics needed for training.

        Examples:
            ```python
            system = ASRSystem(training_config=training_config)
            system.collect_stats()
            # Writes stats under training_config.stats_dir.
            ```
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        logger.info(
            "Collecting stats | exp_dir=%s stats_dir=%s",
            getattr(self.training_config, "exp_dir", None),
            getattr(self.training_config, "stats_dir", None),
        )
        return collect_stats(self._stage_config(self.training_config))

    def train(self, *args, **kwargs):
        """Train the system model.

        Examples:
            ```python
            system = ASRSystem(training_config=training_config)
            system.train()
            # Runs model.fit() and saves the resolved config under exp_dir.
            ```
        """
        self._reject_stage_args("train", args, kwargs)
        model_target = None
        if self.training_config is not None and hasattr(self.training_config, "model"):
            model_config = self.training_config.model
            if isinstance(model_config, DictConfig):
                model_target = model_config.get("_target_")
        logger.info(
            "Training start | exp_dir=%s model=%s",
            getattr(self.training_config, "exp_dir", None),
            model_target or "<unknown>",
        )
        return train(self._stage_config(self.training_config))

    def infer(self, *args, **kwargs):
        """Run inference on the configured datasets.

        Examples:
            ```python
            system = ASRSystem(inference_config=inference_config)
            system.infer()
            # Writes decoded hypotheses under inference_config.inference_dir.
            ```
        """
        self._reject_stage_args("infer", args, kwargs)
        logger.info(
            "Inference start | inference_dir=%s",
            getattr(self.inference_config, "inference_dir", None),
        )
        return infer(self._stage_config(self.inference_config))

    def measure(self, *args, **kwargs):
        """Compute evaluation metrics from hypothesis/reference outputs.

        Examples:
            ```python
            system = ASRSystem(
                metrics_config=metrics_config, inference_config=inference_config
            )
            result = system.measure()
            # result holds the computed metric values.
            ```
        """
        self._reject_stage_args("measure", args, kwargs)
        logger.info(
            "Metrics start | metrics_config=%s",
            self.metrics_config is not None,
        )
        result = measure(
            self._stage_config(self.metrics_config),
            inference_config=self._stage_config(self.inference_config),
        )
        logger.info("results: %s", result)
        return result

    # ---------------------------------------------------------
    # Publication stages (optional overrides)
    # ---------------------------------------------------------
    def pack_model(self, *args, **kwargs):
        """Pack model artifacts into an espnet3 bundle.

        Examples:
            ```python
            system = ASRSystem(
                training_config=training_config,
                publication_config=publication_config,
            )
            system.pack_model()
            # Packs model artifacts under publication_config.pack_model.out_dir.
            ```
        """
        self._reject_stage_args("pack_model", args, kwargs)
        return _pack_model(
            training_config=self._stage_config(self.training_config),
            publication_config=self._stage_config(self.publication_config),
            inference_config=self._stage_config(self.inference_config),
            metrics_config=self._stage_config(self.metrics_config),
        )

    def upload_model(self, *args, **kwargs):
        """Upload model bundle to HuggingFace."""
        self._reject_stage_args("upload_model", args, kwargs)
        return _upload_model(self)

    def pack_demo(self, *args, **kwargs):
        """Pack demo assets into a runnable demo directory."""
        self._reject_stage_args("pack_demo", args, kwargs)
        return _pack_demo(self)

    def upload_demo(self, *args, **kwargs):
        """Upload demo bundle to HuggingFace."""
        self._reject_stage_args("upload_demo", args, kwargs)
        return _upload_demo(self)


# __init_subclass__ only fires for subclasses; check BaseSystem's own
# declaration here so it is held to the same contract.
check_stage_contract(BaseSystem)
