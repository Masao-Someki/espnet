---
title: 📘 Parallel Processing in ESPnet3 (`espnet3.parallel`)
author:
  name: "Masao Someki"
date: 202-07-01
---

This module provides a unified abstraction for parallel and distributed processing across various environments using [Dask](https://docs.dask.org). It is used internally in ESPnet3 for inference, statistics collection, and other batch computations, but can also be used standalone in external applications.

---

## 🚀 Key Features

- Supports multiple execution environments:
  - Local (CPU or GPU)
  - SLURM / PBS / LSF / HTCondor / SSH clusters
  - Kubernetes
- Easy to switch between environments by config
- Minimal boilerplate using context-managed Dask clients
- Utilities for:
  - `parallel_map()`: parallel map across iterable
  - `parallel_submit()`: submit single function calls
  - `parallel_scatter()`: distribute shared data
- Plugin interface to initialize per-worker state (e.g., model loading)

---

## 📦 API Overview

| Function | Description |
|----------|-------------|
| `set_parallel(config)` | Set the global Dask config for parallel execution |
| `get_client(config, plugin)` | Context manager for client with optional `WorkerPlugin` |
| `parallel_map(func, data)` | Apply a function to an iterable in parallel |
| `parallel_submit(func, *args, **kwargs)` | Submit a single task for asynchronous execution |
| `parallel_scatter(data)` | Distribute data to workers for shared access |

---

## 🧩 Usage Examples

### 1. Basic Parallel Map

```python
from omegaconf import OmegaConf
from espnet3.parallel import set_parallel, parallel_map

config = OmegaConf.create({
    "env": "local",
    "n_workers": 4,
    "options": {}
})
set_parallel(config)

def square(x):
    return x * x

results = parallel_map(square, range(10))
print(results)
````

---

### 2. Using a WorkerPlugin (for model initialization)

```python
from dask.distributed import WorkerPlugin, get_worker

class MyPlugin(WorkerPlugin):
    def setup(self, worker):
        worker.model = load_model()  # custom logic

def infer(x):
    worker = get_worker()
    return worker.model.predict(x)

with get_client(plugin=MyPlugin()) as client:
    results = client.map(infer, data)
```

---

### 3. Scattering Shared Data

```python
with get_client() as client:
    shared_data = client.scatter(large_dict)
    results = client.map(lambda x: process(x, shared_data), inputs)
```

> ❗️Note: `scatter()` does not replace `WorkerPlugin`. Use `scatter` for data; use `WorkerPlugin` for one-time per-worker setup.

---

## 🧠 In ESPnet3 Recipes

This module is widely used in ESPnet3 recipes:

* **Inference (`InferenceRunner`)**

  * Uses `parallel_map()` and `WorkerPlugin` to run inference across datasets
* **Statistics Collection (`collect_stats.py`)**

  * Uses `parallel_map()` and `scatter()` to compute feature statistics in parallel
* **Evaluate Scripts**

  * Load `parallel` settings from YAML like below:

```yaml
parallel:
  env: slurm
  n_workers: 16
  options:
    queue: cpu
    memory: 8GB
    cores: 1
    job_extra_directives:
      - "--cpus-per-task=4"
      - "--output=parallel_log/%j.log"
```

---

## ⚙ Supported Environments

| `env` value                 | Backend                                |
| --------------------------- | -------------------------------------- |
| `local`                     | CPU using `LocalCluster`               |
| `local_gpu`                 | GPU using `dask_cuda.LocalCUDACluster` |
| `slurm`, `pbs`, `lsf`, etc. | HPC via `dask_jobqueue`                |
| `ssh`                       | Multi-node SSH-based cluster           |
| `kube`                      | Kubernetes with `dask_kubernetes`      |

> ⚠️ You must install the corresponding package (e.g., `dask_cuda`, `dask_jobqueue`) depending on the environment.

---

## 🛠 Tips and Best Practices

* Use `WorkerPlugin` if you need to load models or configure GPUs per worker.
* Use `scatter()` for broadcasting large shared objects (e.g., tokenizers, config).
* Avoid mixing `client.submit()` and `client.map()` unless you know the implications on task scheduling.
* Use `with get_client() as client` to ensure clean cluster shutdown.

---

## 🧪 Advanced Use Case: Custom Dask Setup

You can instantiate and manage Dask clients directly with your own configuration:

```python
from espnet3.parallel import make_client

config = OmegaConf.create({
    "env": "ssh",
    "n_workers": 4,
    "options": {
        "hosts": ["worker1", "worker2"],
        "connect_options": {"username": "user"},
        ...
    }
})

client = make_client(config)
```

---

## 🧾 License & Attribution

This module wraps [Dask Distributed](https://distributed.dask.org), and is integrated into ESPnet3 under the same Apache 2.0 license.

---

## 🔚 Summary

The `espnet3.parallel` module provides a lightweight, highly flexible abstraction layer over Dask, enabling scalable parallelism in research and production code. It’s designed to work both inside ESPnet and as a general-purpose parallel utility.
