"""Callbacks for ESPnet3 trainer."""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from typeguard import typechecked

logger = logging.getLogger(__name__)


@typechecked
class AverageCheckpointsCallback(Callback):
    """A custom callback for weight averaging over the top-K checkpoints.

    This can be useful to smooth out fluctuations in weights across the best-performing
    models and can lead to improved generalization performance at inference time.

    Behavior:
        - Loads the state_dict from each of the top-K checkpoints saved by given
          ModelCheckpoint callbacks.
        - Averages the model parameters (keys starting with `model.`).
        - Ignores or simply accumulates integer-type parameters
          (e.g., BatchNorm's `num_batches_tracked`).
        - Saves the averaged model as a `.pth` file in `output_dir`.

    Args:
        output_dir (str or Path):
            The directory where the averaged model will be saved.
        best_ckpt_callbacks (List[ModelCheckpoint]):
            A list of ModelCheckpoint callbacks whose top-K checkpoints will be used
            for averaging. Each callback must have `best_k_models` populated.

    Notes:
        - Only keys that start with `model.` are included in the averaging.
        - The final filename will be:
            `{monitor_name}.ave_{K}best.pth`
        - This callback only runs on the global rank 0 process
            (for distributed training).

    Example:
        >>> avg_ckpt_cb = AverageCheckpointsCallback(
        ...     output_dir="checkpoints/",
        ...     best_ckpt_callbacks=[val_loss_ckpt_cb, acc_ckpt_cb]
        ... )
        >>> trainer = Trainer(callbacks=[avg_ckpt_cb])
    """

    def __init__(self, output_dir, best_ckpt_callbacks):
        """Initialize AverageCheckpointsCallback object."""
        self.output_dir = output_dir
        self.best_ckpt_callbacks = best_ckpt_callbacks

    def __repr__(self):
        callbacks = ", ".join(
            self._describe_callback(cb) for cb in self.best_ckpt_callbacks
        )
        return (
            f"{self.__class__.__name__}("
            f"output_dir={self.output_dir!r}, "
            f"best_ckpt_callbacks=[{callbacks}]"
            f")"
        )

    __str__ = __repr__

    @staticmethod
    def _describe_callback(callback):
        return (
            f"{callback.__class__.__name__}("
            f"monitor={getattr(callback, 'monitor', None)!r}, "
            f"save_top_k={getattr(callback, 'save_top_k', None)!r}, "
            f"mode={getattr(callback, 'mode', None)!r}"
            f")"
        )

    def on_validation_end(self, trainer, pl_module):
        """At the end of validation, average the top-K checkpoints and save."""
        if trainer.is_global_zero:
            logger.info("AverageCheckpointsCallback started: %s", self)
            for ckpt_callback in self.best_ckpt_callbacks:
                checkpoints = list(ckpt_callback.best_k_models.keys())
                if not checkpoints:
                    logger.info(
                        "No checkpoints to average for monitor=%r",
                        getattr(ckpt_callback, "monitor", None),
                    )
                    continue

                logger.info(
                    "Averaging %d checkpoints for monitor=%r",
                    len(checkpoints),
                    getattr(ckpt_callback, "monitor", None),
                )
                logger.info("Checkpoint list: %s", checkpoints)

                avg_state_dict = None
                reference_keys = None
                for ckpt_path in checkpoints:
                    logger.info("Loading checkpoint: %s", ckpt_path)
                    state_dict = torch.load(
                        ckpt_path, map_location="cpu", weights_only=False
                    )

                    # for deepspeed checkpoints
                    if "module" in state_dict:
                        state_dict = state_dict["module"]
                    # for PytorchLightning checkpoints
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]

                    if avg_state_dict is None:
                        avg_state_dict = state_dict
                        reference_keys = set(state_dict.keys())
                        logger.info(
                            "Reference keys established: %d keys",
                            len(reference_keys),
                        )
                    else:
                        # Check key consistency
                        current_keys = set(state_dict.keys())
                        if current_keys != reference_keys:
                            raise KeyError(
                                f"Mismatch in keys between checkpoints.\n"
                                f"Expected: {reference_keys}\n"
                                f"Got: {current_keys} (from {ckpt_path})"
                            )
                        for k in avg_state_dict:
                            avg_state_dict[k] = avg_state_dict[k] + state_dict[k]

                int_keys = []
                for k in avg_state_dict:
                    if str(avg_state_dict[k].dtype).startswith("torch.int"):
                        # For int type, not averaged, but only accumulated.
                        # e.g. BatchNorm.num_batches_tracked
                        # (If there are any cases that requires averaging
                        #  or the other reducing method, e.g. max/min, for integer type,
                        #  please report.)
                        int_keys.append(k)
                        pass
                    else:
                        avg_state_dict[k] = avg_state_dict[k] / len(checkpoints)

                if int_keys:
                    logger.info(
                        "Integer parameters only accumulated (not averaged): %d keys",
                        len(int_keys),
                    )
                    logger.info("Integer parameter keys: %s", int_keys)

                # remove extra prefix in model keys
                new_avg_state_dict = {
                    k.removeprefix("model."): v
                    for k, v in avg_state_dict.items()
                    if k.startswith("model.")
                }

                avg_ckpt_path = Path(self.output_dir) / (
                    f"{ckpt_callback.monitor.replace('/', '.')}."
                    + f"ave_{len(checkpoints)}best.pth"
                )
                torch.save(new_avg_state_dict, avg_ckpt_path)
                logger.info("Saved averaged checkpoint: %s", avg_ckpt_path)


class IntervalMetricsLogger(Callback):
    """A custom callback for logging metrics at fixed step intervals.

    This is intended to provide concise, periodic snapshots of training/validation
    metrics in a single line, making it easier to track progress in long runs.

    Behavior:
        - Logs scalar entries from ``trainer.callback_metrics``.
        - Emits logs at every ``log_interval`` steps (skips step 0).
        - Formats values in espnet-2's logging style.
        - Skips non-scalar tensors and non-numeric values, reporting skipped keys.
        - Logs at ``on_train_batch_end`` and ``on_validation_end``.

    Args:
        log_interval (int):
            Number of steps between log outputs. Disabled if <= 0.

    Notes:
        - Step ranges are computed as:
            ``start = max(1, global_step - log_interval + 1)``,
            ``end = global_step``.
        - The log line format is:
            ``{epoch}epoch:{stage}:{start}-{end}batch: key=value, ...``
        - This uses ``trainer.callback_metrics`` for consistency with Lightning's
          callback metric aggregation.

    Example:
        >>> interval_logger = IntervalMetricsLogger(log_interval=100)
        >>> trainer = Trainer(callbacks=[interval_logger, ...])
    """

    def __init__(self, log_interval: int = 100):
        """Initialize the interval logger.

        Args:
            log_interval: Number of steps between log outputs. Disabled if <= 0.
        """
        self.log_interval = int(log_interval)

    def _should_log(self, trainer: L.Trainer) -> bool:
        """Return True when the current step hits the logging interval."""
        step = trainer.global_step
        return step % self.log_interval == 0

    def __repr__(self):
        """Return a readable representation for logging."""
        return f"{self.__class__.__name__}(log_interval={self.log_interval!r})"

    __str__ = __repr__

    def _format_metric(self, value: float) -> str:
        """Format scalar values with fixed or scientific notation."""
        abs_value = abs(value)
        if abs_value != 0 and (abs_value >= 1e4 or abs_value < 1e-3):
            return f"{value:.3e}"
        return f"{value:.3f}"

    def _log_dict(self, trainer: L.Trainer, metrics: dict, stage: str):
        """Log scalar metrics for a given stage if the interval hits."""
        if not metrics or not self._should_log(trainer):
            return

        formatted = {}
        skipped = []
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    skipped.append(key)
                    continue
            try:
                formatted[key] = float(value)
            except (TypeError, ValueError):
                skipped.append(key)

        if formatted:
            end_step = trainer.global_step
            start_step = max(1, end_step - self.log_interval + 1)
            items = ", ".join(
                f"{key}={self._format_metric(formatted[key])}"
                for key in formatted
            )
            logger.info(
                "%depoch:%s:%d-%dbatch: %s",
                trainer.current_epoch,
                stage,
                start_step,
                end_step,
                items,
            )
        if skipped:
            logger.info(
                "IntervalMetricsLogger skipped non-scalar metrics: %s",
                ", ".join(sorted(skipped)),
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ): 
        """Log train metrics after each batch when the interval hits."""
        self._log_dict(trainer, trainer.callback_metrics, stage="train")

    def on_validation_end(self, trainer, pl_module):
        """Log validation metrics at the end of validation when interval hits."""
        self._log_dict(trainer, trainer.callback_metrics, stage="valid")


@typechecked
def get_default_callbacks(
    expdir: str = "./exp",
    log_interval: int = 500,
    best_model_criterion: Union[List[Tuple[str, int, str]], List[List]] = [
        ("valid/loss", 3, "min")
    ],
) -> List[Callback]:
    """Return a list of callbacks tailored for most training workflows.

    Includes:
        - `ModelCheckpoint` for saving the last model checkpoint (`save_last`)
        - One or more `ModelCheckpoint`s for saving the top-K checkpoints according to
            specific metrics
        - `AverageCheckpointsCallback` to compute and save the average model from top-K
            checkpoints
        - `LearningRateMonitor` to track and log learning rates during training
        - `TQDMProgressBar` to show a rich progress bar during training

    Args:
        expdir (str): Directory to store checkpoints and logs.
        log_interval (int): Frequency (in training steps) to refresh the progress bar.
        best_model_criterion (List[Tuple[str, int, str]]): A list of criteria for
            saving top-K checkpoints.
            Each item is a tuple: (metric_name, top_k, mode), where:
            - `metric_name` (str): The name of the validation metric to monitor
                (e.g., "val/loss").
            - `top_k` (int): Number of best models to keep.
            - `mode` (str): "min" to keep models with lowest metric, "max" for highest.

    Returns:
        List[Callback]: A list of callbacks to be passed to the PyTorch Lightning
            Trainer.

    Example:
        >>> from default_callbacks import get_default_callbacks
        >>> callbacks = get_default_callbacks(
        ...     expdir="./exp",
        ...     log_interval=100,
        ...     best_model_criterion=[("val/loss", 5, "min"), ("val/acc", 3, "max")]
        ... )
        >>> trainer = Trainer(callbacks=callbacks, ...)
    """
    last_ckpt_callback = ModelCheckpoint(
        dirpath=expdir,
        save_last="link",
        filename="step{step}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_weights_only=False,
    )

    best_ckpt_callbacks = []
    for monitor, nbest, mode in best_model_criterion:
        best_ckpt_callbacks.append(
            ModelCheckpoint(
                save_top_k=nbest,
                monitor=monitor,
                mode=mode,  # "min" or "max"
                dirpath=expdir,
                save_last=False,
                # Add monitor to filename to avoid overwriting
                # when multiple metrics are used
                filename="epoch{epoch}_step{step}_" + monitor.replace("/", "."),
                auto_insert_metric_name=False,
                save_on_train_epoch_end=False,
                save_weights_only=True,
                enable_version_counter=False,  # just overwrite
            )
        )
    ave_ckpt_callback = AverageCheckpointsCallback(
        output_dir=expdir, best_ckpt_callbacks=best_ckpt_callbacks
    )

    # Monitor learning rate
    lr_callback = LearningRateMonitor()

    # Interval metrics logger
    interval_metrics_logger = IntervalMetricsLogger(log_interval=log_interval)

    # Progress bar
    progress_bar_callback = TQDMProgressBar(
        refresh_rate=log_interval,
        leave=True
    )

    callbacks = [
        last_ckpt_callback,
        *best_ckpt_callbacks,  # unpack list to add them to the list of callbacks.
        ave_ckpt_callback,
        interval_metrics_logger,
        lr_callback,
        progress_bar_callback,
    ]
    logger.info("Default callbacks initialized: %s", callbacks)
    return callbacks
