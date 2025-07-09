---
title: ESPnet3 Callbacks for PyTorch Lightning
author:
  name: "Masao Someki"
date: 202-07-01
---

This module provides a collection of callbacks tailored for **integrating ESPnet functionality into PyTorch Lightning** training workflows. These callbacks enhance model checkpointing, learning rate tracking, progress reporting, and model weight averaging—key features required for robust and reproducible training in ESPnet3.

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [AverageCheckpointsCallback](#averagecheckpointscallback)

   * [Purpose](#purpose)
   * [Behavior](#behavior)
   * [Limitations](#limitations)
3. [get\_default\_callbacks Function](#get_default_callbacks-function)

   * [Included Callbacks](#included-callbacks)
   * [Arguments](#arguments)
   * [Example Usage](#example-usage)
4. [Configuration via YAML](#configuration-via-yaml)
5. [Integration with ESPnet3LightningTrainer](#integration-with-espnet3lightningtrainer)
6. [Notes](#notes)

---

## Overview

The callbacks defined in `callback.py` are specifically designed to extend PyTorch Lightning with functionality required by ESPnet3. These include:

* Top-K model checkpointing based on multiple metrics
* Averaging of top-performing checkpoints
* Learning rate monitoring
* Rich progress visualization
* Seamless integration with Hydra-based configuration

---

## AverageCheckpointsCallback

### Purpose

`AverageCheckpointsCallback` is a custom PyTorch Lightning callback that averages the weights of the top-K checkpoints (based on specified metrics) at the end of training. This is particularly useful for improving generalization by smoothing the fluctuations of individual checkpoints.

### Behavior

* Loads `state_dict`s from top-K checkpoints defined by one or more `ModelCheckpoint` callbacks.
* Averages all model parameters with prefix `model.`.
* Skips averaging for integer-type tensors (e.g., `num_batches_tracked` in BatchNorm); these are summed instead.
* Saves the averaged model to a `.pth` file under the specified `output_dir`.

#### File Naming

The output filename is generated as:

```
{monitor_name}.ave_{K}best.pth
```

Example:

```
valid.loss.ave_3best.pth
```

### Notes

* Only runs on global rank 0 (for distributed training).
* Will raise an error if checkpoint `state_dict` keys do not match exactly.
* Only keys with `model.` prefix are included in the final saved checkpoint.

---

## `get_default_callbacks` Function

This utility function returns a standardized list of PyTorch Lightning callbacks commonly used in ESPnet3 training workflows.

### Included Callbacks

* **Last Checkpoint Callback** (`ModelCheckpoint` with `save_last`)
* **Top-K Model Checkpoints** (per metric via `ModelCheckpoint`)
* **Model Averaging** (`AverageCheckpointsCallback`)
* **Learning Rate Monitoring** (`LearningRateMonitor`)
* **Progress Bar** (`TQDMProgressBar`)

### Arguments

| Argument               | Type                         | Description                                                                                                                                                                      |
| ---------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `expdir`               | `str`                        | Directory for storing checkpoints and logs                                                                                                                                       |
| `log_interval`         | `int`                        | Interval (in training steps) to refresh the progress bar                                                                                                                         |
| `best_model_criterion` | `List[Tuple[str, int, str]]` | List of criteria for top-K model saving. Each tuple contains: <br>• `metric_name` (e.g. `"valid/loss"`) <br>• `top_k`: number of models to keep <br>• `mode`: `"min"` or `"max"` |

### Example Usage

```python
from callbacks import get_default_callbacks

callbacks = get_default_callbacks(
    expdir="./exp",
    log_interval=100,
    best_model_criterion=[
        ("valid/loss", 5, "min"),
        ("valid/acc", 3, "max")
    ]
)
```

---

## Configuration via YAML

You can define custom callbacks in your configuration files using Hydra-compatible syntax. These user-defined callbacks will be appended to the default ones.

```yaml
trainer:
  callbacks:
    - _target_: lightning.pytorch.callbacks.EarlyStopping
      monitor: valid/loss
      patience: 5
      mode: min

    - _target_: lightning.pytorch.callbacks.StochasticWeightAveraging
      swa_lrs: 1e-3
```

> ⚠️ Avoid naming conflicts with default callbacks such as `ModelCheckpoint` and `AverageCheckpointsCallback`.

---

## Integration with `ESPnet3LightningTrainer`

The `ESPnet3LightningTrainer` class automatically initializes and applies these callbacks during setup:

* It uses `get_default_callbacks()` internally.
* You may optionally add user-defined callbacks via the `trainer.callbacks` field in the configuration.
* The callbacks are passed directly to PyTorch Lightning's `Trainer`.

---

## Notes

* The averaged model checkpoint (`*.ave_*best.pth`) is a plain PyTorch model and can be loaded using `torch.load()`.
* `AverageCheckpointsCallback` is especially recommended when training with multiple validation metrics.
* The callback system is designed to be extensible and Hydra-compatible for modular research workflows.
