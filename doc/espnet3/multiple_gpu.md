## ESPnet3: Multi-GPU and Multi-Node Execution

ESPnet3 relies on PyTorch Lightning for distributed training and on the
Provider/Runner abstraction for scalable inference or data processing.  The same
configuration runs locally, with multiple GPUs in a single job, or across SLURM
clusters without modifying the Python code.

---

### 1. Training with PyTorch Lightning

Distributed training is configured directly in the experiment YAML file.  The
example below launches a data-parallel job on two nodes with four GPUs per node.

```yaml
trainer:
  accelerator: gpu
  devices: 4
  num_nodes: 2
  strategy: ddp
  precision: 16-mixed
  gradient_clip_val: 1.0
  log_every_n_steps: 200
```

Lightning handles process spawning, communication, gradient accumulation, and
checkpointing.  No wrapper scripts are required—`espnet3.trainer.trainer` reads
this configuration and forwards it to Lightning.

When running under a scheduler (e.g., SLURM) make sure the submission command
requests matching resources, for example:

```bash
sbatch --nodes=2 --gres=gpu:4 --cpus-per-task=8 train.sh
```

The training script itself is unchanged between local and cluster runs.

---

### 2. Inference or evaluation with runners

For multi-GPU inference, decoding, or scoring jobs ESPnet3 provides the
`BaseRunner` class.  A runner processes indices in parallel while an
`EnvironmentProvider` constructs datasets and models on each worker.

```python
from espnet3.runner.inference_provider import InferenceProvider
from espnet3.runner.base_runner import BaseRunner
from espnet3.parallel import set_parallel

class DecodeProvider(InferenceProvider):
    @staticmethod
    def build_dataset(cfg):
        return load_eval_split(cfg.dataset)

    @staticmethod
    def build_model(cfg):
        model = build_model(cfg.model)
        return model.to(cfg.model.device)

class DecodeRunner(BaseRunner):
    @staticmethod
    def forward(idx: int, *, dataset, model, beam_size=5):
        sample = dataset[idx]
        return {
            "utt_id": sample["utt_id"],
            "hyp": model.decode(sample, beam_size=beam_size),
        }

provider = DecodeProvider(cfg, params={"beam_size": 8})
runner = DecodeRunner(provider)

set_parallel(cfg.parallel_gpu)  # same config works locally or on SLURM
results = runner(range(len(eval_set)))
```

Workers automatically receive their own dataset/model instances.  To bind one
GPU per worker specify `env: slurm` (or any Dask JobQueue backend) and use
`job_extra_directives` such as `--gres=gpu:1` in the parallel configuration.

---

### 3. Local vs. cluster performance

The refactored runner supports three modes—local, synchronous cluster jobs, and
asynchronous cluster submissions.  The same decoding runner was benchmarked in
[#6178](https://github.com/espnet/espnet/pull/6178#issuecomment-3400164353) on an
A40 GPU with the OWSM-V4 medium (1B) model over 1,000 LibriSpeech test-clean
utterances:

| Environment  | #GPUs | Wall time (s) |
| ------------ | ----- | ------------- |
| local        | 1     | 1220          |
| local        | 2     | 695           |
| slurm / sync | 1     | 1336          |
| slurm / sync | 2     | 669           |
| slurm / sync | 4     | 369           |

In synchronous mode the driver waits for all workers to finish, whereas in
asynchronous mode the submission script becomes a lightweight dispatcher and the
worker jobs continue even if the driver exits early.

---

By combining Lightning for training and the Provider/Runner API for inference,
ESPnet3 offers a uniform interface for single-GPU experiments, multi-GPU servers,
and large-scale clusters.
