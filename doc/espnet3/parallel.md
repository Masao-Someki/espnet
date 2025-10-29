## ESPnet3 Parallel Execution Guide

ESPnet3 offers two complementary layers for distributed work:

1. **Provider/Runner API** – recommended for most workloads.  It builds the
   environment on each worker and executes a static `forward` function.
2. **`espnet3.parallel` utilities** – lower-level helpers that expose Dask
   primitives directly when you need custom scheduling logic.

Both layers are backed by Dask JobQueue and therefore support local
multiprocessing, SLURM, PBS, LSF, and other cluster managers.

---

### 1. Recommended path: Providers and runners

Define a provider to instantiate datasets/models and a runner that processes the
indices of interest.  Switching from local execution to cluster execution is as
simple as providing a `parallel` section in the configuration and calling
`set_parallel` before invoking the runner.

```yaml
parallel_gpu:
  env: slurm
  n_workers: 4
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 32GB
    walltime: 02:00:00
    job_extra_directives:
      - "--gres=gpu:1"
```

```python
from espnet3.parallel import set_parallel

set_parallel(cfg.parallel_gpu)
results = runner(range(num_items))  # local, sync, or async depending on config
```

The runner automatically chooses between three execution modes:

- **local** – runs sequentially on the driver when no parallel config is set.
- **synchronous** – uses a shared Dask client; results are returned to the
  caller in order.
- **asynchronous** – enabled with `async_mode=True`.  Each shard is written to a
  JSON specification and submitted via Dask JobQueue.  Workers can keep running
  even if the driver exits, mirroring the behaviour showcased in
  [#6178](https://github.com/espnet/espnet/pull/6178#issuecomment-3400164353).

#### Customising submission commands

`BaseRunner` exposes a hook that rewrites the job submission command emitted by
Dask JobQueue.  During asynchronous runs the runner injects a subclass of the
cluster’s `job_cls` so you can tweak the shell command, add environment
variables, or replace the prologue entirely.  This allows advanced users to
change the `sbatch` invocation without forking ESPnet3.

---

### 2. Low-level access: `espnet3.parallel`

For experiments that need direct control over Dask you can interact with the
underlying utilities yourself.  `make_client`, `parallel_for`, and
`wrap_func_with_worker_env` provide the same building blocks used by the
Provider/Runner layer.

```python
from espnet3.parallel import get_client, parallel_for
from espnet3.parallel.parallel import DictReturnWorkerPlugin

with get_client(cfg.parallel_cpu) as client:
    plugin = DictReturnWorkerPlugin(setup_fn=make_env)
    client.register_worker_plugin(plugin, name="env")

    for item in parallel_for(user_func, dataset_indices, client=client):
        process(item)
```

Use this level when you need to stitch ESPnet3 together with external systems or
when you want full control over task submission and result handling.

---

### 3. Why prefer runners?

- Configuration symmetry: the same YAML selects local, synchronous, or
  asynchronous execution.
- Automatic environment construction: datasets, models, and helpers are built on
  each worker via the provider.
- Extensibility: because runners are regular Python classes you can still drop
  to the lower-level API when necessary without rewriting your preparation or
  inference logic.

In short, start with providers and runners for most projects and reach for the
`espnet3.parallel` module when you need fine-grained control over the Dask
cluster or its submission scripts.
