---
title: ESPnet3 Inference Stage
author:
  - name: "Masao Someki"
  - name: "Elias Naske"
date: 2026-05-21
---

# ESPnet3 Inference Stage

The `infer` stage runs model inference on the provided test set(s) and writes the outputs to disk.
This file is used to measure model performance in the [`measure`](./measure.md) stage.

## Run
When executing a recipe with `run.py`, specify `infer` as an argument of the `--stages` flag.

```bash
python run.py --stages infer --inference_config conf/inference.yaml
```

## Configuration

Keep the core settings in `inference.yaml`. For the full list, see
[Inference configuration](../config/infer_config.md).

| Config section  | Required | Description                                   |
| --------------- | -------- | --------------------------------------------- |
| `model`         | ✅        | model to run inference with                   |
| `dataset`       | ✅        | definition of the test set                    |
| `inference_dir` | ✅        | root output location                          |
| `input_key`     |          | dataset field or fields passed into the model |
| `output_fn`     |          | function used to format the output files      |
| `parallel`      |          | local or distributed runner settings          |

## Outputs

Inference writes one directory per test set:

```text
<inference_dir>/
└── <test_name>/
    ├── hyp.scp
    └── ...
```

The filenames are determined by:

- `output_keys` when it is set
- otherwise the keys returned by `output_fn` for the first sample, excluding
  `idx_key`

### SCP Files

In the `.scp` format, each line represents an utterance and takes the following form:

```text
{utt_id} {value}
```

The value is determined by `output_fn` and can be either:
- a scalar value (`str`, `int`, `float`, `bool`)
- a non-scalar value (e.g. `dict`, `numpy.ndarray`, `torch.tensor`)

Scalar values are written directly into SCP files.
Non-scalar values are written as artifacts to a file, and the SCP stores a path to said file.
Artifacts are written under:
```text
<inference_dir>/<test_name>/<field_name>/
```
The file type depends on the return type of `output_fn`:

| Value type          | Default artifact type | Saved as |
| ------------------- | --------------------- | -------- |
| `dict`              | `json`                | `.json`  |
| `numpy.ndarray`     | `npy`                 | `.npy`   |
| CPU `torch.Tensor`  | `npy`                 | `.npy`   |
| other Python object | `pickle`              | `.pkl`   |

The config can also be set to force other types, such as `wav`.
E.g., if `output_fn` returns:

```python
{
    "utt_id": "utt1",
    "audio": wav_numpy,
}
```

and `inference.yaml` contains:

```yaml
output_artifacts:
  audio:
    type: wav
    sample_rate: 16000
```

then inference writes:

```text
<inference_dir>/
└── <test_name>/
    ├── audio.scp
    └── audio/
        ├── utt1.wav
        └── utt2.wav
```

and `audio.scp` stores the generated `.wav` paths.


### Custom artifact writers

If you want to save a custom type such as PNG, add a writer function and point
to it from config.

Example config:

```yaml
output_artifacts:
  image:
    writer:
      _target_: src.inference.write_png_artifact
```

Example function:

```python
from pathlib import Path


def write_png_artifact(*, value, output_path):
    path = Path(output_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    value.save(path)
    return path
```

The writer must return the written path. That path is stored in the SCP file.


## Implementation Details
### Inference Providers and Runners

Inference is implemented as a Provider/Runner loop.

The provider is responsible for:

- building the dataset for the active test set
- instantiating the model
- exposing config-derived runtime parameters

The runner is responsible for:

- pulling one sample or one batch from the dataset
- calling the model with the configured `input_key`
- normalizing the result through `output_fn`
- returning values that can be written into SCP files

Conceptually:
```python
provider = InferenceProvider(config)
runner = InferenceRunner(provider=provider, async_mode=False)
results = runner(range(len(provider.build_dataset(config))))
```

### `output_fn`

`output_fn` is called right after the model returns.

If provided, `output_fn` is called as:

```python
output_fn(data=data, model_output=model_output, idx=idx)
```

It should return a dict for a single sample, or a list of dicts for batched
inference.

Typical output:

```python
{
    "utt_id": "utt1",
    "hyp": "hello world",
}
```

The base runner accepts either a single index or a list of indices. That is why
`output_fn` must be able to handle:

- a single sample plus scalar `idx`
- or batched input where `data` is a list and `idx` is a list

Minimal single-sample example:

```python
def build_output(*, data, model_output, idx):
    return {
        "utt_id": data["uttid"],
        "hyp": model_output["text"],
        "ref": data.get("text", ""),
    }
```


## Using a custom model

There a two common paths when using a custom models:

1. keep `InferenceRunner` and replace only `model` and `output_fn`
2. replace `InferenceRunner` when the normal flow is not enough

It is generally recommended to keep `InferenceRunner` and only replace it for special use cases. 

### Example: Custom Decoding Algorithm

This is the common case:

- you want to keep the same dataset
- you want to keep the same SCP writing path
- but you want your own decoding algorithm

In that case, keep the default runner and replace only `model` and `output_fn`.

Example `inference.yaml`:

```yaml
dataset:
  test:
    - name: test
      data_src: mini_an4/asr
      data_src_args:
        split: test

model:
  _target_: src.inference.MyGreedyDecoder
  checkpoint_path: ${exp_dir}/last.ckpt
  beam_size: 1

input_key: speech
output_fn: src.inference.build_output

provider:
  _target_: espnet3.systems.base.inference_provider.InferenceProvider
runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner
```

Example `src/inference.py`:

```python
from pathlib import Path

import torch


class MyGreedyDecoder:
    def __init__(self, checkpoint_path, beam_size=1):
        self.checkpoint_path = Path(checkpoint_path)
        self.beam_size = beam_size
        self.model = self._load_model()

    def _load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model = checkpoint["model"]
        model.eval()
        return model

    def __call__(self, speech):
        tokens = self.model.decode(speech, beam_size=self.beam_size)
        text = self.model.tokenizer.decode(tokens)
        return {"text": text, "tokens": tokens}


def build_output(*, data, model_output, idx):
    return {
        "utt_id": data.get("uttid", str(idx)),
        "hyp": model_output["text"],
        "token_ids": " ".join(str(v) for v in model_output["tokens"]),
        "ref": data.get("text", ""),
    }
```

The runtime order is:

1. `InferenceRunner` loads one sample from the dataset
2. it calls `model(**inputs)`
3. it calls `build_output(...)`
4. it writes `hyp.scp`, `token_ids.scp`, and `ref.scp`

### When to replace `InferenceRunner`

Replace the runner only when `model -> output_fn -> SCP` is not enough.

Examples:

- streaming decode with internal state
- multi-step search with custom batching
- non-standard output validation

Minimal custom runner example:

```python
from espnet3.systems.base.inference_runner import InferenceRunner


class MyInferenceRunner(InferenceRunner):
    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        data = dataset[idx]
        model_output = model.decode_stream(data["speech"])
        return {
            "utt_id": data.get("uttid", str(idx)),
            "hyp": model_output["text"],
            "ref": data.get("text", ""),
        }
```

Config:

```yaml
runner:
  _target_: src.inference.MyInferenceRunner
```

Use this path only when `output_fn` is not enough.

## Related pages

- [Inference configuration](../config/infer_config.md)
- [Measure stage](./measure.md)
- [Provider / runner](../core/parallel/provider_runner.md)
