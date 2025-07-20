
---
title: 📘 Parallel Processing in ESPnet3 (`espnet3.parallel`)
author:
  name: "Masao Someki"
date: 2025-07-17
---

This module provides a unified abstraction for parallel and distributed processing across multiple compute environments using [Dask](https://dask.org). It powers scalable inference, statistics collection, and batch operations in ESPnet3, and can be used independently.

---

## 🚀 Key Features

- ✅ Support for:
  - Local (CPU / GPU)
  - HPC clusters (SLURM, PBS, LSF, HTCondor, SSH)
  - Kubernetes
- ✅ Minimal boilerplate with `get_client()` context manager
- ✅ Per-worker environment via `setup_fn`
- ✅ Streaming-like task consumption with `parallel_for()`
- ✅ Utility wrappers: `parallel_map()`, `parallel_submit()`

---

## 📦 API Summary

| Function                        | Description                                         |
|--------------------------------|-----------------------------------------------------|
| `set_parallel(config)`         | Set global Dask config                              |
| `get_client(config, setup_fn)` | Context-managed Dask client                         |
| `parallel_map(func, data)`     | Parallel map over an iterable                       |
| `parallel_for(func, args)`     | Stream-like parallel execution                      |

---

## 🧩 Usage Examples

### 1. Basic Parallel Map (CPU)

```python
from omegaconf import OmegaConf
from espnet3.parallel import set_parallel, parallel_map

config = OmegaConf.create({
    "env": "local",
    "n_workers": 4,
    "options": {}
})
set_parallel(config)

def square(x): return x * x

results = parallel_map(square, range(10))
print(results)
```

---

### 2. Injecting Per-Worker Environment with `setup_fn`

```python
from espnet3.parallel import get_client, wrap_func_with_worker_env

def setup_fn():
    model = load_model()
    dataset = load_dataset()
    return {"model": model, "dataset": dataset}

def infer(idx, model, dataset):
    sample = dataset[idx]
    return model.predict(sample)

with get_client(setup_fn=setup_fn) as client:
    results = client.map(inference_fn, inputs)

```

---

### 3. Using `parallel_for` for Streaming

```python
from espnet3.parallel import parallel_for

def compute(x):
    return x * 2

for result in parallel_for(compute, range(100)):
    print("Got result:", result)

# You can also run with setup function
with get_client(setup_fn=setup_fn) as client:
    for result in parallel_for(compute, range(100)):
        print("Got result:", result)

```

> ⚠ `parallel_for()` yields results as they complete. Good for large, slow tasks.

---

## ⚙ Supported Environments

| `env` value          | Backend                            |
| -------------------- | ---------------------------------- |
| `local`              | CPU (Dask `LocalCluster`)          |
| `local_gpu`          | GPU (`dask_cuda.LocalCUDACluster`) |
| `slurm`, `pbs`, etc. | HPC via `dask_jobqueue`            |
| `ssh`                | SSH multi-node cluster             |
| `kube`               | Kubernetes (`dask_kubernetes`)     |

> 🔧 You must install the relevant backend packages (e.g., `dask_cuda`, `dask_kubernetes`, `dask_jobqueue`) as needed.

---

## 🛠 Best Practices

* ✅ Use `setup_fn()` to load large models or configs once per worker.
  * It is not recommended to send data as argument.
* 🧹 Use `with get_client():` to ensure clean client shutdown.

---

## 🔬 Advanced: Manual Setup

```python
from espnet3.parallel import set_parallel
from omegaconf import OmegaConf

config = OmegaConf.create({
    "env": "slurm",
    "n_workers": 4,
    "options": {
        "queue": "gpu",
        "cores": 1,
        "processes": 1,
        "memory": 16GB,
        "walltime": "30:00", # Wait time to run setup_fn
        "job_extra_directives": {
          "--gres=gpu:1",
          "--ntasks-per-node=1",
          "--cpus-per-task=4",
          "--output=logs/%j-%x.log",
        }
    }
})

set_parallel(config)
```

---

## 🧾 License

ESPnet3 uses [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). This module builds on [Dask Distributed](https://distributed.dask.org) and [Dask Jobqueue](https://jobqueue.dask.org/).
