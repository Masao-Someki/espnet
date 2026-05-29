---
title: ESPnet3 Measure Stage
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-22
---

# ESPnet3 Measure Stage

This page describes the model evaluation flow in ESPnet3.

## 1. Run

```bash
python run.py --stages measure --metrics_config conf/metrics.yaml
```

## 2. Outputs

The summary file format is:

```text
<inference_dir>/
├── metrics.json
└── test-clean/
    ├── ref.scp
    ├── hyp.scp
    ├── wer_alignment
    └── cer_alignment
```

`metrics.json` is keyed by metric class path, then by test set name.
`measure()` resolves test sets in this order:

1. If `metrics_config.dataset.test` exists, use each item's `name`.
2. Otherwise, scan `metrics_config.inference_dir` for subdirectories.

## 3. Configuration

The `measure` stage is configured using `metrics.yaml`
This config file defines the metrics that are used to evaluate the model.

Each entry in `metrics.yaml` is handled like this:

1. instantiate `metrics_config.metrics[*].metric`
2. resolve input SCP paths for one `test_name`
3. call the metric class

Example config for WER:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER # import path for the function
      ref_key: ref
      hyp_key: hyp
```

Here, the values for `ref_key` and `hyp_key` are the names of the SCP file created during the `infer` stage.

`measure()` will instantiate the class provided in `_target_` and pass the arguments as follows:

```python
{
    "ref": Path("exp/.../inference/<test_name>/ref.scp"),
    "hyp": Path("exp/.../inference/<test_name>/hyp.scp"),
}
```
where `<test_name>` is the name of the test partition (e.g., `test-clean`).

> [!IMPORTANT]
> `measure()` does not preload SCP contents into lists. It resolves file paths and passes them directly to each metric.

### Inputs and SCP filenames

Each metric can receive inputs in two ways.

If `inputs` is defined in config:

```yaml
metrics:
  - metric:
      _target_: my_pkg.metrics.MyMetric
    inputs:
      ref: ref
      hyp: hyp
      prompt: prompt
```

then ESPnet3 resolves:

- `data["ref"] -> <test_name>/ref.scp`
- `data["hyp"] -> <test_name>/hyp.scp`
- `data["prompt"] -> <test_name>/prompt.scp`

If `inputs` is omitted, `measure()` falls back to the metric instance's
`ref_key` and `hyp_key`.


### Sample config

```yaml
recipe_dir: .
exp_tag:
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      ref_key: ref
      hyp_key: hyp
      clean_types:

  - metric:
      _target_: espnet3.systems.asr.metrics.cer.CER
      ref_key: ref
      hyp_key: hyp
      clean_types:
```


## 4. Custom Metrics

To implement a custom metric, create a class that inferits from `espnet3.components.metrics.base_metric.BaseMetric` and define the following methods:
```python
class MyMetric(BaseMetric):

  def __init__(
    self,
    **keys # The keys for the SCP file paths (`ref_key` and `hyp_key` in the example above)
  ) -> None
  # This function typically saves the provided keys as attributes
  # so that they can be accessed in __call__()

  def __call__(
    self,
    data: Dict[str, Path], # Paths to the SCP files
    test_name: str, # Current test set name
    output_dir: str, # The root of `inference_dir`
  ) -> Dict[str, Any] # Mapping the metric name to a value, e.g., {"WER": 0.05}
```

Because `__call__()` takes file paths as input, this means the metric class itself reads SCP contents.
For aligned SCP inputs, the normal implementation pattern is
`BaseMetric.iter_inputs(...)`.



## Related pages

- [Metrics configuration](../config/measure_config.md)
- [Custom metrics](../core/components/metrics.md)
