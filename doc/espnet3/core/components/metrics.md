---
title: ESPnet3 Metrics
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Metrics

This page describes how to implement custom metrics for the current ESPnet3
measurement flow.

The base class is:

- `espnet3.components.metrics.base_metric.BaseMetric`

The stage entrypoint is:

- `espnet3.systems.base.metric.measure`

::: note
If you want the full file flow from `metrics.yaml` to `metrics.json`, use the
interactive explorer below.
This page focuses on the metric component contract itself.
:::

<MetricsExplorer />

## Metric contract

Metrics are instantiated from `metrics_config.metrics[*].metric`. Each metric is
called as:

```python
metric(data: Dict[str, Path], test_name: str, output_dir: Path)
```

This is intentionally path-based.
ESPnet3 resolves the requested inputs to SCP paths, but reading and alignment
happen inside the metric.

Typical input:

```python
{
    "ref": Path(".../ref.scp"),
    "hyp": Path(".../hyp.scp"),
}
```

`output_dir` is passed through as `metrics_config.inference_dir` — in a
typical Hydra/OmegaConf config this is a plain `str`, not a `Path` instance,
even though the abstract signature is typed as `Path`. Wrap it yourself (as
the built-in `WER`/`CER`/`TER` metrics do: `Path(output_dir) / test_name`)
rather than relying on it already being a `Path`.

**`metrics_config.inference_dir` must be set for `measure()` to run at all.**
It is populated when you pass `--inference_config` alongside
`--metrics_config` (the shared run copies `inference_config.inference_dir`
into `metrics_config`); passing only `--training_config` +
`--metrics_config` does not populate it, so set `inference_dir` explicitly in
`metrics.yaml` if you run `measure` standalone.

## BaseMetric

```python
from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric


class MyMetric(BaseMetric):
    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        output_dir: Path,
    ) -> Dict[str, float]:
        ...
```

Return values must be JSON-serializable because `measure()` stores them in
`<inference_dir>/metrics.json`.

## iter_inputs() helper

For aligned SCP text inputs, use:

```python
for utt_id, row in self.iter_inputs(data, "ref", "hyp"):
    ref = row["ref"]
    hyp = row["hyp"]
```

`iter_inputs()`:

- opens the requested SCP files
- reads them in file order
- checks that utterance IDs match
- yields one aligned row at a time

This is the normal pattern for text metrics such as WER and CER.

Metrics that work directly from file paths can ignore `iter_inputs()` and read
`data[...]` themselves.

## Config aliases

`measure()` accepts either:

- implicit keys via the metric's `ref_key` / `hyp_key`
- explicit aliases through `inputs`

Example:

```yaml
metrics:
  - metric:
      _target_: my_pkg.metrics.CustomMetric
    inputs:
      ref: text
      hyp: hyp
      prompt: prompt
```

This means `measure()` passes:

- `data["ref"]` -> `text.scp`
- `data["hyp"]` -> `hyp.scp`
- `data["prompt"]` -> `prompt.scp`

## Multiple entries of the same metric class

`measure()` keys its results dict by the metric's fully-qualified class path
(e.g. `espnet3.systems.asr.metrics.wer.WER`). If you configure the same class
twice in `metrics.yaml` (for example, `WER` once against `hyp.scp` and again
against `hyp_nbest0.scp` with different `inputs`), the second entry overwrites
the first under that same key in `metrics.json`, and both write to the same
alignment filename (e.g. `wer_alignment`). Give each entry a distinguishable
metric class (a small subclass is enough) if you need more than one scored
entry of the same metric type to survive.

## TER and `bpemodel`

`TER` needs a trained SentencePiece model to tokenize text before scoring, so
point `bpemodel` at the path your recipe's tokenizer stage actually wrote to —
typically `${tokenizer.save_path}/bpe.model`, which (per the TEMPLATE
tokenizer config) resolves to something like `${data_dir}/bpe_5000/bpe.model`,
not an `exp/` path.

## Example: text metric

```python
from pathlib import Path
from typing import Dict

import jiwer

from espnet3.components.metrics.base_metric import BaseMetric


class SimpleWER(BaseMetric):
    def __init__(self, ref_key: str = "ref", hyp_key: str = "hyp") -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        output_dir: Path,
    ) -> Dict[str, float]:
        refs = []
        hyps = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            refs.append(row[self.ref_key])
            hyps.append(row[self.hyp_key])
        return {"WER": jiwer.wer(refs, hyps) * 100}
```

## Example: path-driven metric

```python
from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric


class FileCountMetric(BaseMetric):
    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        output_dir: Path,
    ) -> Dict[str, float]:
        with open(data["hyp"], encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        return {"num_hypotheses": float(count)}
```

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Metrics configuration"
    desc="See how metric classes and inputs are selected from YAML."
    icon="tabler:settings-2"
    href="../../stages/metrics.html"
  />
  <DocCard
    title="Metrics stage"
    desc="Return to the stage-level metrics flow."
    icon="tabler:route"
    href="../../stages/metrics.html"
  />
  <DocCard
    title="Components overview"
    desc="Return to the full component map."
    icon="tabler:puzzle"
    href="./index.html"
  />
</DocCards>
