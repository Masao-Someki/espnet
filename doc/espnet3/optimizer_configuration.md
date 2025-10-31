## ESPnet3: Optimiser and Scheduler Configuration

ESPnet3 wraps PyTorch Lightning so that optimisers and schedulers can be defined
purely from the Hydra configuration.  Two modes are supported by
`espnet3.trainer.model.LitESPnetModel.configure_optimizers`:

1. a single optimiser/scheduler pair (`optim` + `scheduler`)
2. multiple optimiser/scheduler pairs (`optims` + `schedulers`)

The sections below describe both.

---

### 1. Single optimiser

Use `optim` and `scheduler` when the entire model shares one optimiser.  The
entries are passed directly to `hydra.utils.instantiate`, so any optimiser or
scheduler from PyTorch (or a custom class) is supported.

```yaml
optim:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1.0e-2

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100000
```

No additional wiring is necessary—ESPnet3 instantiates both objects, attaches
the scheduler to the optimiser, and returns them to Lightning.

---

### 2. Multiple optimisers

When different parts of the model need their own optimiser, switch to `optims`
and `schedulers`.  Each entry contains a nested `optim` block and a `params`
string that selects parameters whose names contain the substring.

```yaml
optims:
  - optim:
      _target_: torch.optim.Adam
      lr: 0.0005
    params: encoder
  - optim:
      _target_: torch.optim.Adam
      lr: 0.001
    params: decoder

schedulers:
  - scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      step_size: 5
      gamma: 0.5
  - scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      step_size: 10
      gamma: 0.1
```

Important rules enforced by `configure_optimizers`:

- Do not mix the single- and multi-optimiser modes.  Either use `optim` +
  `scheduler` or `optims` + `schedulers`.
- Every optimiser entry must include `params` and `optim`.
- Each trainable parameter must match exactly one optimiser.  ESPnet3 raises an
  error if parameters are missing or assigned twice.
- The number of scheduler entries must equal the number of optimisers.  They are
  matched by position, so the first scheduler controls the first optimiser, etc.

Under the hood ESPnet3 wraps the instantiated optimisers with
`MultipleOptim`/`MultipleScheduler` so that Lightning sees them as a single
optimiser while still stepping all underlying objects together.
