---
title: ESPnet3 Create Dataset Stage
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-14
---

# ESPnet3 Create Dataset Stage

`create_dataset()` is responsible for downloading and preparing datasets.
For each unique dataset source defined for a partition (`dataset.train`, `dataset.valid`, and `dataset.test` in `training.yaml`), it:

1. resolves the dataset module
2. instantiates a [builder](#builder) object.
3. prepares source files, if needed
4. build the dataset, if needed

The same dataset source is only prepared once per stage run.

## Run

When executing a recipe with `run.py`, specify `create_dataset` as an argument of the `--stages` flag.

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
```

## Where the code lives

Within a typical recipe structure, the necessary files are organized like so:

```text
egs3/<recipe>/<task>/
├── conf/
│   └── training.yaml   # config file
└── dataset/
    ├── __init__.py     # exports Dataset and Builder classes
    ├── builder.py      # handles source preparation and build-time side effects
    └── dataset.py      # defines the Dataset class used for training and inference.
```

## Configuration 

The `create_dataset` stage is configured from the `create_dataset` block in
`training.yaml`.

Example:

```yaml
dataset_dir: ${recipe_dir}/data/mini_an4

create_dataset:
  recipe_dir: ${recipe_dir}
  dataset_dir: ${dataset_dir}

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
  train:
    - name: train
      data_src: mini_an4/asr
      data_src_args:
        split: train
        data_path: ${dataset_dir}
  valid:
    - name: valid
      data_src: mini_an4/asr
      data_src_args:
        split: valid
        data_path: ${dataset_dir}
```

The following keys are available:

| Key           | Description       | Example                           |
| ------------- | ----------------- | --------------------------------- |
| `recipe_dir`  | Recipe directory  | `egs3/mini_an4/asr`               |
| `dataset_dir` | Dataset directory | `egs3/mini_an4/asr/data/mini_an4` |


## Builder

The code for preparding the dataset is defined in a builder class in `builder.py`, which inherits from [`espnet3.components.data.DatasetBuilder`](../../../espnet3/components/data/dataset_builder.py).

The builder has two main responsibilities:
- **Source Preparation**: download, extract, validate, or locate raw assets
- **Building**: run task-ready preprocessing or other recipe-local dumping data

To implement these, the builder class must define the following methods:

| Method                         | Returns | What it does                                                                                            |
| ------------------------------ | ------- | ------------------------------------------------------------------------------------------------------- |
| `is_source_prepared(**kwargs)` | `bool`  | Checks if the raw dataset source files are already available. If `True`, `prepare_source()` is skipped. |
| `prepare_source(**kwargs)`     | `None`  | Prepares raw source files.                                                                              |
| `is_built(**kwargs)`           | `bool`  | Checks if manifest files are already built. If `True`, `build()` is skipped.                            |
| `build(**kwargs)`              | `None`  | Builds manifest files for each partition, preprocesses audio files, etc.                                |

The arguments passed to the builder methods come from the `create_dataset`
block in `training.yaml`.
For example, if the config looks like this:
```yaml
create_dataset:
  recipe_dir: ${recipe_dir}
  source_dir: ${dataset_dir}
```

The builder will be called like this:

```python
if not builder.is_source_prepared(recipe_dir=..., source_dir=...):
  builder.prepare_source(recipe_dir=..., source_dir=...)

if not builder.is_built(recipe_dir=..., source_dir=...):
  builder.build(recipe_dir=..., source_dir=...)
```

## `dataset/__init__.py`

To ensure that the dataset and builder classes are accessible to other modules, they should be exported in `dataset/__init__.py` as `Dataset` and `DatasetBuilder` respectively.

Minimal example:

```python
from egs3.my_recipe.asr.dataset.builder import MyDatasetBuilder as DatasetBuilder
from egs3.my_recipe.asr.dataset.dataset import MyDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
```

## How dataset modules are resolved

Dataset resolution is shared with the normal dataset loading path:

1. `data_src: mini_an4/asr`
2. `data_src: egs3.mini_an4.asr.dataset`
3. omitted `data_src`, which loads `${recipe_dir}/dataset/__init__.py`

Details are in:

- [Dataset references and builders](../core/components/datasets.md)

## Example: `mini_an4`

`egs3/mini_an4/asr/dataset/builder.py` is a full build example.

Behavior:

- `prepare_source()` extracts the AN4 archive under the recipe dataset area
- `build()` converts audio and writes manifest TSVs under `data/manifest/`

The resulting tree is roughly:

```text
egs3/mini_an4/asr/
└── data/
    ├── manifest/
    │   ├── train.tsv
    │   ├── valid.tsv
    │   └── test.tsv
    └── wav/
        ├── train/
        └── test/
```

Minimal conceptual export in `__init__.py`:

```python
from egs3.mini_an4.asr.dataset.builder import MiniAn4Builder as DatasetBuilder
from egs3.mini_an4.asr.dataset.dataset import MiniAn4Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
```

## Example: `librispeech_100`

`egs3/librispeech_100/asr/dataset/builder.py` is the contrasting pattern.

Behavior:

- `prepare_source()` only validates that the LibriSpeech tree exists
- `is_built()` simply reuses source readiness
- `build()` is effectively a no-op validation path

This recipe reads the original corpus layout directly instead of generating
separate manifests.

This is the contrasting pattern to `mini_an4`: the builder still participates
in the stage lifecycle, but the recipe chooses not to materialize a separate
manifest representation.

## Notes

- `create_dataset` should be deterministic and safe to re-run
- source preparation and build are intentionally separate checks
- the same dataset source is only prepared once even if it appears in multiple
  splits

## Related pages

- [Dataset references and builders](../core/components/datasets.md)
- [DataOrganizer](../core/components/data-organizer.md)
- [Training dataset config](./train/dataset.md)
