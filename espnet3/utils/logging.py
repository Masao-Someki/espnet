from __future__ import annotations

import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

from omegaconf import DictConfig, ListConfig, OmegaConf

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_ENV_KEYS = [
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONPATH",
    "PL_GLOBAL_SEED",
    "PL_TORCH_DISTRIBUTED_BACKEND",
]


def _coerce_level(level: int | str | None) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level_upper = level.upper()
        if level_upper in logging._nameToLevel:  # type: ignore[attr-defined]
            return logging._nameToLevel[level_upper]  # type: ignore[attr-defined]
    raise ValueError(f"Invalid logging level: {level}")


def _to_dict(config: object | None) -> MutableMapping[str, object]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    if isinstance(config, (dict, Mapping)):
        return dict(config)
    if isinstance(config, Namespace):
        return vars(config)
    raise TypeError(
        "Logging configuration must be a mapping-like object, "
        f"got: {type(config)!r}"
    )


def configure_logging(
    output_path: str | Path | None = None,
    *,
    level: int | str | None = None,
    to_stdout: bool = True,
    overwrite: bool = True,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_LOG_DATE_FORMAT,
) -> logging.Logger:
    resolved_level = _coerce_level(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    # Remove existing handlers so repeated invocations replace configuration.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handlers: list[logging.Handler] = []
    if to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handlers.append(stream_handler)

    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            path,
            mode="w" if overwrite else "a",
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handlers.append(file_handler)

    if not handlers:
        # Fallback to stdout if no handler has been configured.
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handlers.append(stream_handler)

    for handler in handlers:
        root_logger.addHandler(handler)

    logging.captureWarnings(True)

    return logging.getLogger("espnet3")


def setup_logging_from_config(
    config: object | None,
) -> tuple[logging.Logger, MutableMapping[str, object]]:
    options = {}
    if config is not None:
        options = _to_dict(config)

    output_path = (
        options.get("path")
        or options.get("output_path")
        or options.get("file")
        or options.get("filepath")
    )

    logger = configure_logging(
        output_path=output_path,
        level=options.get("level"),
        to_stdout=options.get("stdout", True),
        overwrite=options.get("overwrite", True),
        fmt=options.get("format", DEFAULT_LOG_FORMAT),
        datefmt=options.get("date_format", DEFAULT_LOG_DATE_FORMAT),
    )

    if output_path:
        options["path"] = str(Path(output_path).expanduser())
    else:
        options["path"] = None

    return logger, options


def log_environment_snapshot(
    logger: logging.Logger,
    *,
    env_keys: Iterable[str] | None = None,
    log_all_env_vars: bool = False,
) -> None:
    logger.info("===== Environment Snapshot =====")
    logger.info("Command line: %s", " ".join(sys.argv))
    logger.info("Current working directory: %s", Path.cwd())
    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version.replace("\n", " "))
    try:
        import platform

        logger.info("Platform: %s", platform.platform())
    except Exception as exc:  # pragma: no cover - platform failure is unlikely
        logger.debug("Failed to retrieve platform information: %s", exc)

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_devices = torch.cuda.device_count() if cuda_available else 0
        logger.info(
            "PyTorch: version=%s cuda_available=%s num_cuda_devices=%s",
            torch.__version__,
            cuda_available,
            cuda_devices,
        )
    except Exception as exc:  # pragma: no cover - torch may be optional in tests
        logger.debug("PyTorch information unavailable: %s", exc)

    try:
        import lightning

        logger.info("Lightning: version=%s", lightning.__version__)
    except Exception as exc:  # pragma: no cover - lightning may be optional in tests
        logger.debug("Lightning information unavailable: %s", exc)

    if log_all_env_vars:
        env_items = dict(os.environ)
        logger.info(
            "Logging all environment variables (%d entries)...", len(env_items)
        )
    else:
        requested_keys = list(dict.fromkeys((env_keys or []) + DEFAULT_ENV_KEYS))
        env_items = {
            key: os.environ[key]
            for key in requested_keys
            if key in os.environ
        }
        logger.info(
            "Logging selected environment variables (%d/%d entries found)...",
            len(env_items),
            len(requested_keys),
        )

    if not env_items:
        logger.info("No matching environment variables were found.")
        return

    for key in sorted(env_items):
        logger.info("ENV[%s]=%s", key, env_items[key])


def log_configuration(
    logger: logging.Logger,
    config: object,
    *,
    header: str = "Resolved Configuration",
) -> None:
    logger.info("===== %s =====", header)

    if isinstance(config, (DictConfig, ListConfig)):
        try:
            config_text = OmegaConf.to_yaml(config, resolve=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to convert OmegaConf to YAML: %s", exc)
            config_text = repr(config)
    else:
        config_text = repr(config)

    for line in config_text.splitlines():
        logger.info("%s", line)


def _count_parameters(module) -> tuple[int, int]:
    try:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    except Exception:
        return 0, 0
    return total, trainable


def log_model_summary(
    logger: logging.Logger,
    model,
    *,
    header: str = "Model Summary",
) -> None:
    logger.info("===== %s =====", header)

    try:  # pragma: no cover - summary import is optional
        from lightning.pytorch.utilities.model_summary import ModelSummary

        summary = ModelSummary(model, max_depth=2)
        logger.info("%s", summary)
    except Exception as exc:
        logger.warning("Model summary unavailable: %s", exc)
        logger.info("Model repr: %s", model)

    total_params, trainable_params = _count_parameters(model)
    if total_params or trainable_params:
        logger.info(
            "Parameters: total=%d trainable=%d",
            total_params,
            trainable_params,
        )

    base_model = getattr(model, "model", None)
    if base_model is not None and base_model is not model:
        logger.info(
            "Base model class: %s.%s",
            base_model.__class__.__module__,
            base_model.__class__.__qualname__,
        )
        base_total, base_trainable = _count_parameters(base_model)
        if base_total or base_trainable:
            logger.info(
                "Base parameters: total=%d trainable=%d",
                base_total,
                base_trainable,
            )
        logger.info("Base model structure:\n%s", base_model)


def log_dataset_snapshot(
    logger: logging.Logger,
    *,
    train_dataset=None,
    valid_dataset=None,
) -> None:
    datasets = [
        ("train", train_dataset),
        ("valid", valid_dataset),
    ]
    for split, dataset in datasets:
        if dataset is None:
            continue
        logger.info(
            "%s dataset class: %s.%s",
            split.capitalize(),
            dataset.__class__.__module__,
            dataset.__class__.__qualname__,
        )
        try:
            size = len(dataset)
        except Exception as exc:
            logger.warning(
                "Unable to determine %s dataset size: %s",
                split,
                exc,
            )
        else:
            logger.info("%s dataset size: %s", split.capitalize(), size)


def _format_callback(callback) -> str:
    try:
        return repr(callback)
    except Exception:
        return f"<callback {callback.__class__.__module__}.{callback.__class__.__qualname__}>"


def log_trainer_summary(
    logger: logging.Logger,
    trainer,
    *,
    header: str = "Trainer Summary",
) -> None:
    trainer_obj = getattr(trainer, "trainer", trainer)
    logger.info("===== %s =====", header)
    logger.info(
        "Trainer class: %s.%s",
        trainer_obj.__class__.__module__,
        trainer_obj.__class__.__qualname__,
    )
    logger.info("Trainer repr:\n%s", trainer_obj)

    accelerator = getattr(trainer_obj, "accelerator", None)
    if accelerator is not None:
        logger.info("Accelerator: %s", accelerator)
    strategy = getattr(trainer_obj, "strategy", None)
    if strategy is not None:
        logger.info("Strategy: %s", strategy)
    precision = getattr(trainer_obj, "precision", None)
    if precision is not None:
        logger.info("Precision: %s", precision)

    default_root_dir = getattr(trainer_obj, "default_root_dir", None)
    if default_root_dir:
        logger.info(
            "Trainer default_root_dir: %s",
            Path(default_root_dir).expanduser().resolve(),
        )

    callbacks = list(getattr(trainer_obj, "callbacks", []) or [])
    if callbacks:
        logger.info("Registered callbacks (%d):", len(callbacks))
        for callback in callbacks:
            logger.info("  %s", _format_callback(callback))
    else:
        logger.info("Registered callbacks: none")

    loggers = getattr(trainer_obj, "loggers", None)
    if loggers:
        logger.info("Lightning loggers (%d):", len(loggers))
        for lg in loggers:
            logger.info("  %s", _format_callback(lg))
    else:
        single_logger = getattr(trainer_obj, "logger", None)
        if single_logger:
            logger.info("Lightning logger: %s", _format_callback(single_logger))
        else:
            logger.info("Lightning logger: none")

    checkpoint_paths = set()
    for callback in callbacks:
        dirpath = getattr(callback, "dirpath", None)
        if dirpath:
            checkpoint_paths.add(Path(dirpath).expanduser())
    checkpoint_callback = getattr(trainer_obj, "checkpoint_callback", None)
    if checkpoint_callback is not None:
        dirpath = getattr(checkpoint_callback, "dirpath", None)
        if dirpath:
            checkpoint_paths.add(Path(dirpath).expanduser())

    if checkpoint_paths:
        logger.info("Checkpoint directories:")
        for path in sorted(checkpoint_paths):
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                resolved = path
            logger.info("  %s", resolved)
    elif default_root_dir:
        fallback = Path(default_root_dir).expanduser() / "checkpoints"
        logger.info("Checkpoint directory (default): %s", fallback.resolve())
    else:
        logger.info("Checkpoint directory: unavailable")


def log_experiment_directories(
    logger: logging.Logger,
    *,
    expdir: str | Path | None = None,
    statsdir: str | Path | None = None,
    decode_dir: str | Path | None = None,
) -> None:
    if expdir:
        exp_path = Path(expdir).expanduser()
        logger.info("Experiment directory: %s", exp_path.resolve())
        logger.info("Experiment checkpoint directory: %s", (exp_path / "checkpoints").resolve())
    if statsdir:
        stats_path = Path(statsdir).expanduser()
        logger.info("Stats directory: %s", stats_path.resolve())
    if decode_dir:
        decode_path = Path(decode_dir).expanduser()
        logger.info("Decode directory: %s", decode_path.resolve())
