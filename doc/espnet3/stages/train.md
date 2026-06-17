---
title: ESPnet3 Train Stage
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-15
---

# ESPnet3 Train Stage

The `train` stage runs model training using a [PyTorch Lightning trainer](../core/components/trainer.md) based on the dataset and hyperparameters defined in `training.yaml` and saves model checkpoints and logs.

## 1. Run

```bash
python run.py --stages train --training_config conf/training.yaml
```

## 2. Configuration

Training is configured in `training.yaml` using the sections shown in the table below.
For a detailed list of options, see [Training Configuration](../config/train_config.md) and the links in the table.


| Section                    | Description                                 | Details                                                                                    |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `task`                     | task entrypoint for ESPnet2-style models    |
| `model`                    | model definition and normalization settings |
| `dataset`                  | `train` and `valid` splits                  | [Data Organizer](../core/components/data-organizer.md)                                     |
| `dataloader`               | collate and iterator settings               | [Dataloader + Collate](../core/components/dataloader.md)                                   |
| `trainer`                  | Lightning trainer configuration             | [Trainer](../core/components/trainer.md)                                                   |
| `optimizer`, `scheduler`   | single-optimizer training path              | [Optimizer + Scheduler](../core/components/optimizer_configuration.md)                     |
| `optimizers`, `schedulers` | named multi-optimizer path                  | [Multiple Optimizers and Schedulers](../core/components/multiple_optimizers_schedulers.md) |
| `exp_dir`                  | training output directory                   |


## 3. Outputs

Training outputs are written under `exp_dir`, including:

- Checkpoints
- Logs
- (If configured) TensorBoard output

## Related pages
<DocCards :cols="3">
  <DocCard
    title="Training configuration"
    desc="See all options for cofiguring the train stage."
    icon="tabler:file-code"
    href="../config/train_config.html"
  />
  <DocCard
    title="Inference stage"
    desc="Information on the inference stage."
    icon="tabler:puzzle"
    href="./inference.html"
  />
  <DocCard
    title="Trainer"
    desc="Information about the trainer component"
    icon="tabler:tool"
    href="../core/components/trainer.html"
  />
</DocCards>