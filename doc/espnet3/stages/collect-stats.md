---
title: ESPnet3 Collect Stats Stage
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-14
---

# ESPnet3 Collect Stats Stage

`collect_stats` computes shape files and feature statistics used by later
training steps.
This serves two broad purposes:

1. **Shape information for batching**: Precomputing feature lengths lets the iterator adjust batches based on sequence size, which is one of the main ways ESPnet avoids out-of-memory errors.
2. **Statistics for normalization**: Certain forms of normalization, such as global mean and variance normalization, need dataset-level statistics to be computed. Computing these once here allows them to be reused later.

This information is saved to the following files:

```text
${stats_dir}/
├── train/
│   ├── feats_shape       # features shapes for batching
│   ├── feats_stats.npz   # features statistics for normalization
│   └── stats_keys
└── valid/
    ├── feats_shape
    ├── feats_stats.npz
    └── stats_keys
```

Note that `collect_stats` only processes the dataset's `train` and `valid` splits; `test` is ignored.

## Run

When executing a recipe with `run.py`, specify `collect_stats` as an argument of the `--stages` flag.

```bash
python run.py --stages collect_stats --training_config conf/training.yaml
```

## Model Requirements

The model must possess a `collect_feats()` method.
An implementation of this method exists by default for all models built on ESPnet tasks (e.g. ASR, TTS).

Custom models should provide a compatible implementation of the method with the following interface:

```collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]```

For features of variable length, the return dictionary should contain a matching `*_lengths` tensor containing the lengths for each sample in the feature tensor.

Example:
```python
class MyCustomModel
  def collect_feats(
      self,
      speech: torch.Tensor,
      speech_lengths: torch.Tensor,
      **kwargs,
  ) -> Dict[str, torch.Tensor]:
      feats, feats_lengths = self._extract_feats(speech, speech_lengths)
      return {"feats": feats, "feats_lengths": feats_lengths}
```

## Configuration

The `collect_stats` stage is configured in the same `training.yaml` used for training.

At minimum, the `stats_dir` key must be set to the directionary where the [output files](#outputs) will be dumped.
Components that require shape or stats files (e.g. `model`, `dataloader`) should point to the corresponding files in `${stats_dir}/train/` or `${stats_dir}/valid/` (Note that these paths are read-only; results are always written to `stats_dir`).

Example:

```yaml
stats_dir: ${exp_dir}/stats

# For shape-based batching
dataloader:
  train:
    iter_factory:
      batches:
        shape_files:
          - ${stats_dir}/train/feats_shape

# For normalization
model:
  normalize: global_mvn
  normalize_conf:
    stats_file: ${stats_dir}/train/feats_stats.npz
```

### Advanced: GPU-based stats collection

If `parallel` is configured in `training.yaml`, `collect_stats` can reuse ESPnet3's
parallel execution helpers for heavier feature extraction workloads.

Example:

```yaml
parallel:
  env: slurm
  n_workers: 8
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 16GB
    walltime: 30:00
    job_extra_directives:
      - "--gres=gpu:1"
```

## Related pages

- [Training config](../config/train_config.md)
- [Dataloader](./train/dataloader.md)
