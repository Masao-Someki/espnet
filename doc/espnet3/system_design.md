## ESPnet3 Design Overview

ESPnet3 emphasises reproducibility, flexibility, and maintainability.  Compared
with ESPnet2 the project leans on Hydra/OmegaConf for configuration and on the
Provider/Runner abstraction for execution.  This section summarises the main
design pillars.

---

### 1. Dataset preparation and sharing

- Promote Hugging Face Datasets for a standardised, shareable format.
- Continue supporting custom manifests (JSON, Kaldi SCP, etc.) when required.
- `ESPnet3Dataset` remains a convenient wrapper when a recipe expects the legacy
  collate behaviour, but using it is optional.

```python
from datasets import load_dataset
from espnet3.data.dataset import ESPnet3Dataset

dataset = ESPnet3Dataset(load_dataset("some_dataset", split="train"))
```

The same provider/runner logic used for inference also powers data preparation,
so large corpora can be re-generated or cleaned consistently across different
compute environments (see [data_preparation.md](./data_preparation.md)).

---

### 2. Dataloader and batching strategy

Speech data varies drastically in length.  ESPnet3 keeps the ESPnet2 tooling for
samplers and collators but exposes them through Hydra-driven configuration.
Lhotse integration enables dynamic batching and on-the-fly feature extraction.

```python
from lhotse.dataset import DynamicBucketingSampler
from espnet3.data.loader import make_dataloader

dataloader = make_dataloader(dataset, sampler=DynamicBucketingSampler(...))
```

---

### 3. Models and trainers

- `LitESPnetModel` wraps ESPnet models so they plug directly into PyTorch
  Lightning.
- `ESPnet3LightningTrainer` delegates training to Lightning while handling ESPnet
  specifics such as checkpoint averaging and statistics collection.
- Configuration is declarative: recipes override the necessary parts of the
  Hydra config instead of editing Python files.

```python
from espnet3.trainer.trainer import ESPnet3LightningTrainer
from espnet3.trainer.model import LitESPnetModel

model = LitESPnetModel(task)
trainer = ESPnet3LightningTrainer(model=model, config=config, expdir="exp")
trainer.fit()
```

---

### 4. Provider/Runner architecture

Based on the refactor described in
[#6178](https://github.com/espnet/espnet/pull/6178#issuecomment-3393671961), the
execution layer is split into **providers** and **runners**:

- A provider builds datasets, models, and helper objects per worker.
- A runner implements a static `forward` method that consumes those objects.

This design has several benefits:

- **Mode parity** – the same runner works locally, with multiprocessing, or on a
  Dask/SLURM cluster.  Switching modes only requires changing the `parallel`
  configuration.
- **Asynchronous submissions** – runners can emit shard specifications and submit
  detached jobs.  Example wall-clock numbers for OWSM-V4 medium (1B) decoding are
  published in [this comment](https://github.com/espnet/espnet/pull/6178#issuecomment-3400164353).
- **Custom job control** – advanced users can tweak the Dask JobQueue submission
  command (e.g., the `sbatch` invocation) through the runner hook without
  modifying ESPnet3 internals.

Providers and runners therefore connect local development workflows with large
cluster deployments seamlessly.

---

### 5. Customisation and extensibility

ESPnet3 keeps the PyTorch Lightning ecosystem available: callbacks, precision
plugins, gradient accumulation strategies, and profiling integrations are all
configured via YAML.  When a project needs functionality beyond the built-in
runners, developers can still rely on the lower-level
[`espnet3.parallel`](./parallel.md) utilities.

---

In summary, ESPnet3 delivers a reproducible, Python-native research stack that
scales from laptops to HPC clusters while keeping recipes concise and
maintainable.
