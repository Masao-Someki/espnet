---
title: ESPnet3 Publication Stages
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-26
---

# ESPnet3 Publication Stages

The publication stage is used to publish a trained model to Hugging Face.

This is a two-step process:

| Step           | Description                       | Implementation                                                                        |
| -------------- | --------------------------------- | ------------------------------------------------------------------------------------- |
| `pack_model`   | Prepares the model files.         | [`espnet3.utils.publish_utils.pack_model`](../../../espnet3/utils/publish_utils.py)   |
| `upload_model` | Uploads the model to Hugging Face | [`espnet3.utils.publish_utils.upload_model`](../../../espnet3/utils/publish_utils.py) |

## 1. Run

```bash
python run.py \
  --stages pack_model upload_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
```

## 2. Configuration

The stage is configured in `conf/publication.yaml`.
The config contains the following sections:

| Section                                        | Description                                      |
| ---------------------------------------------- | ------------------------------------------------ |
| `pack_model.strategy`                          | choose `auto`, `espnet2`, or `espnet3` packing   |
| `pack_model.out_dir`                           | output bundle directory                          |
| `pack_model.decode_dir`                        | directory searched for `scores.json`             |
| `pack_model.readme_template`, `readme_context` | README template and extra template values        |
| `pack_model.include`, `extra`, `exclude`       | extra copy and exclusion control                 |
| `pack_model.include_data_dir`                  | include `training_config.data_dir` in the bundle |
| `pack_model.files`, `yaml_files`               | explicit metadata entries copied into the bundle |
| `pack_model.espnet2`                           | espnet2-specific packing spec                    |
| `upload_model`                                 | Hugging Face upload settings                     |

For a full list of options, see [Publication Configuration](../config/publish_config.md).

## 3. `pack_model`

`pack_model` creates a publishable bundle directory, usually:

```text
<exp_dir>/model_pack
```

### Packing Strategy

Model packing differs between ESPnet2 and ESPnet3.
The appropriate strategy can be set through `pack_model.strategy`, which supports the following:

| Tag              | Description                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| `auto` (default) | use `espnet2` when `training_config.task` is set, otherwise use `espnet3` |
| `espnet2`        | force ESPnet2 packing                                                     |
| `espnet3`        | force ESPnet3 packing                                                     |

### Bundle Directory

Current ESPnet3 packing copies recipe assets such as:

| Asset             | Description                |
| ----------------- | -------------------------- |
| `conf/`           | Configuration files        |
| `src/`            | Source code                |
| `run.py`          | Inference entry point      |
| `pixi.toml`       | Pixi configuration         |
| `pixi.lock`       | Pixi dependency resolution |
| `.python-version` | Python version information |

and usually includes the experiment directory and, when enabled, the recipe
`data_dir`.

`pack_model` writes metadata in `meta.yaml` in the bundle root.
That metadata is later used by tools, such as when [running inference through `InferenceSession`](#packaged-model-inference).

In practice the packed tree often looks like:

```text
model_pack/
├── conf/
├── src/
├── exp/
├── data/         # included if `include_data_dir: true` and configured `data_dir` exists
├── run.py
├── meta.yaml
├── README.md
└── scores.json
```

### Additional file controls

`publication.yaml` can also define additional files to include or exclude using the following fields:

| Key          | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| `include`    | paths copied into the bundle before `extra`                     |
| `extra`      | more paths copied into the bundle                               |
| `exclude`    | patterns skipped while copying                                  |
| `yaml_files` | YAML files, saved to `model_pack/yaml_files/`                   |
| `files`      | other artifacts (e.g checkpoints), saved to `model_pack/files/` |

These allow adding or explicitly registering artifacts in `meta.yaml`.


## 4. Packaged model inference 

Direct packaged-model inference is provided by:

- `espnet3.publication.InferenceSession`

Typical use:

```python
from espnet3.publication import InferenceSession

session = InferenceSession.from_pretrained(
    "espnet/your-model-tag",
    trust_user_code=True,
)
result = session(audio_array)
```

Important behavior:

- it can load `conf/inference.yaml` from the packed bundle
- it can use bundle metadata from `meta.yaml`
- it can enable bundled recipe code such as `src/` when
  `trust_user_code=True`

This is the main reason current espnet3 packing includes recipe configs and
recipe-local user code.

## 5. `upload_model`

`upload_model` uploads the packed directory to Hugging Face.

Required field:

- `publication_config.upload_model.hf_repo`

The packed directory must already exist before upload runs.

## Related pages

- [Publication configuration](../config/publish_config.md)
- [Inference stage](./inference.md)
