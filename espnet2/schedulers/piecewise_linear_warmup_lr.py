"""Piecewise linear warm up learning rate scheduler module."""

from typing import List, Union
from omegaconf import ListConfig

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class PiecewiseLinearWarmupLR(_LRScheduler, AbsBatchStepScheduler):
    """The PiecewiseLinearWarmupLR scheduler

    This scheduler is similar to WarmupLR Scheduler except that
    the warmup stage is piecewise linear.

    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps_list: Union[ListConfig, List[Union[int, float]]] = [0, 25000],
        warmup_lr_list: Union[ListConfig, List[float]] = [0.0, 0.001],
        last_epoch: int = -1,
    ):
        self.warmup_steps_list = warmup_steps_list
        self.warmup_lr_list = warmup_lr_list

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(warmup_steps_list={self.warmup_steps_list}, "
            f"warmup_lr_list={self.warmup_lr_list})"
        )

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            np.interp(
                step_num,
                self.warmup_steps_list,
                self.warmup_lr_list,
                right=lr * self.warmup_steps_list[-1] ** 0.5 * step_num**-0.5,
            )
            for lr in self.base_lrs
        ]
