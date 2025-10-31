## ESPnet3: Data Preparation with Providers and Runners

ESPnet3 replaces the shell-script based pipelines from ESPnet2 with a unified
Python workflow.  Datasets, feature extraction, and data cleaning are performed
through the **Provider/Runner** abstraction that powers both training and
inference.  The same code runs locally on a laptop, on multiprocessing setups,
and on large clusters managed by Dask.

This guide focuses on:

1. Defining providers and runners for data preparation.
2. Executing those runners locally and on clusters.
3. Incorporating data cleaning in the same pipeline.

---

### 1. Defining the environment: `EnvironmentProvider`

A provider encapsulates everything that has to be instantiated once per
process—datasets, tokenisers, models, or utility objects.  For data preparation
we usually construct datasets and lightweight helpers.  Providers receive a
Hydra configuration so the implementation stays declarative.

```python
from espnet3.runner.inference_provider import InferenceProvider
from omegaconf import DictConfig

class PrepProvider(InferenceProvider):
    @staticmethod
    def build_dataset(cfg: DictConfig):
        # Load raw manifests and return a list/sequence of samples.
        return load_raw_samples(cfg.dataset)

    @staticmethod
    def build_model(cfg: DictConfig):
        # Optional: build reusable helpers (feature extractor, normaliser, …).
        return build_feature_pipeline(cfg.model)
```

Anything returned by the provider is injected into the runner as keyword
arguments.  Because providers are serialisable, they work in local, parallel
and asynchronous modes without additional changes.

---

### 2. Implementing the computation: `BaseRunner`

A runner defines a static `forward` method that processes a single index.  The
example below performs feature extraction and includes an optional cleaning
step, showing how preparation and cleaning share the same infrastructure.

```python
from espnet3.runner.base_runner import BaseRunner

class PrepRunner(BaseRunner):
    @staticmethod
    def forward(idx: int, *, dataset, model, cleaner=None, output_dir=None):
        sample = dataset[idx]
        feats = model.extract(sample["audio"], sr=sample["sr"])

        if cleaner is not None:
            sample = cleaner(sample)
            if sample is None:  # filtered out
                return None

        save_features(sample["utt_id"], feats, output_dir)
        return {
            "utt_id": sample["utt_id"],
            "features": feats.shape,
        }
```

Because `forward` is static it is pickle-safe and can be shipped to remote
workers.  Any additional runtime arguments (e.g., `cleaner`, `output_dir`) can
be provided through the provider params or when instantiating the runner.

---

### 3. Running locally

Local execution is the default when no parallel configuration is set.  The
provider builds the environment once and the runner iterates through the
requested indices.

```python
from omegaconf import OmegaConf

cfg = OmegaConf.load("prep.yaml")
provider = PrepProvider(cfg, params={"output_dir": "exp/data"})
runner = PrepRunner(provider)

indices = list(range(len(provider.build_env_local()["dataset"])))
runner(indices)  # returns a list of metadata dictionaries
```

---

### 4. Scaling out with Dask (synchronous mode)

To distribute the same runner across multiple workers, define a `parallel`
section in the YAML configuration and pass it to `set_parallel` before
invocation.  No code changes are required.

```yaml
parallel:
  env: slurm
  n_workers: 8
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 32GB
    walltime: 04:00:00
    job_extra_directives:
      - "--gres=gpu:1"
```

```python
from espnet3.parallel import set_parallel

set_parallel(cfg.parallel)
results = runner(range(total_items))  # transparently runs on the cluster
```

Workers call `PrepProvider.make_worker_setup_fn()` to instantiate their own
copy of the dataset and helpers before executing `forward`.

---

### 5. Asynchronous shards for large jobs

For extremely long jobs you can enable the asynchronous mode introduced in the
runner refactor.  Pass `async_mode=True` to the runner and ESPnet3 will
serialize job specifications, submit them through Dask JobQueue, and optionally
store the per-shard outputs as JSONL files:

```python
async_runner = PrepRunner(
    provider,
    async_mode=True,
    async_specs_dir="./specs",
    async_result_dir="./results",
)
job_meta = async_runner(range(total_items))
```

This mode mirrors the behaviour documented in
[#6178](https://github.com/espnet/espnet/pull/6178#issuecomment-3400164353): the
same runner can be executed locally, in synchronous SLURM jobs, or via detached
submissions without modifying the preparation logic.

---

### 6. Data cleaning within the same pipeline

Since cleaning is just additional Python code inside `forward`, you can:

- Drop samples that fail validation by returning `None`.
- Apply noise-reduction or text normalisation before writing outputs.
- Emit side metadata (e.g., cleaning statistics) from the returned dictionary.

All cleaning happens inside the same Provider/Runner pipeline, so the behaviour
is identical regardless of whether the job runs locally or on a cluster.

---

### 7. Sharing prepared data

Prepared features can be written in any format (JSONL, SCP, Hugging Face
datasets, etc.).  Because preparation is now a regular Python workflow, you can
upload the resulting manifests to Hugging Face or store them on shared storage
without extra tooling.

By structuring data preparation around providers and runners you gain a single
code path that scales from interactive prototyping to thousands of jobs on
HPC clusters.
