## ESPnet3: Callback Mechanisms in Training

ESPnet3 relies on PyTorch Lightning for training orchestration, so the vast
majority of Lightning callbacks are immediately available.  On top of that, the
`espnet3.trainer.callbacks` module ships a curated stack of defaults that are
applied when a recipe calls `get_default_callbacks`.  This document explains the
default behaviour and how to extend it in your own experiments.

---

### Default callback stack

`get_default_callbacks` returns the following callbacks, all pre-configured to
write into the experiment directory:

1. **`ModelCheckpoint` (last)** – keeps the latest checkpoint under
   `${expdir}` so that training can be resumed.
2. **`ModelCheckpoint` (best-k)** – one instance per entry in
   `best_model_criterion`, each tracking a validation metric and keeping the
   top-*k* checkpoints.
3. **`AverageCheckpointsCallback`** – averages the best checkpoints tracked by
   the callbacks above and writes `<metric>.ave_<k>best.pth` once validation
   finishes.
4. **`LearningRateMonitor`** – logs optimiser learning rates so that they are
   visible in TensorBoard or any Lightning-compatible logger.
5. **`TQDMProgressBar`** – provides an interactive progress bar whose refresh
   rate can be controlled from the configuration file.

All of these callbacks live in
[`espnet3/trainer/callbacks.py`](../../espnet3/trainer/callbacks.py) and are
instantiated automatically unless you override the callback list explicitly.

---

### Controlling checkpoint selection

The metrics that drive checkpoint selection are configured through
`best_model_criterion` in the experiment YAML:

```yaml
best_model_criterion:
  - - valid/loss
    - 3
    - min
  - - valid/wer
    - 2
    - min
```

Each item is interpreted as `(monitor, top_k, mode)`.  In this example we keep
three checkpoints with the smallest `valid/loss` and two checkpoints with the
smallest `valid/wer`.  `AverageCheckpointsCallback` will in turn average the
weights tracked by these callbacks.

---

### Adjusting progress logging

The `TQDMProgressBar` refresh interval defaults to 500 steps.  Override the
value by passing `log_interval` when calling `get_default_callbacks` from your
recipe:

```python
from espnet3.trainer.callbacks import get_default_callbacks

callbacks = get_default_callbacks(
    expdir=str(expdir),
    log_interval=50,
    best_model_criterion=[("valid/loss", 5, "min")],
)
```

Because callbacks are regular Python objects, you can append or replace entries
before constructing the Lightning trainer.

---

### Providing custom callbacks through Hydra

Recipes can also instantiate callbacks directly from the YAML configuration via
Hydra/OmegaConf.  Simply disable the default stack and enumerate your desired
callbacks in `trainer.callbacks`:

```yaml
trainer:
  callbacks:
    - _target_: espnet3.trainer.callbacks.AverageCheckpointsCallback
      output_dir: ${expdir}
      best_ckpt_callbacks:
        - _target_: lightning.pytorch.callbacks.ModelCheckpoint
          monitor: valid/cer
          save_top_k: 5
          mode: min
```

Mixing both approaches is perfectly valid—use `get_default_callbacks` for the
common utilities and append any domain-specific callbacks that your project
requires.

---

### ESPnet-specific checkpoint averaging details

`AverageCheckpointsCallback` is an ESPnet-provided extension that runs on the
rank-zero process at the end of validation.  For every `ModelCheckpoint`
callback listed in `best_ckpt_callbacks` it loads the corresponding top-*k*
checkpoints, verifies that the parameter sets match, and averages weights whose
keys start with `model.`.  Integer tensors—such as
`BatchNorm.num_batches_tracked`—are accumulated instead of averaged so the
resulting statistics remain meaningful.  The averaged weights are written to
`${expdir}/${monitor}.ave_<k>best.pth`, regardless of whether the checkpoints
were produced by native PyTorch Lightning, DeepSpeed, or other supported
strategies.

You can reuse the callback outside the defaults by instantiating it directly:

```python
from espnet3.trainer.callbacks import AverageCheckpointsCallback

ave_ckpt = AverageCheckpointsCallback(
    output_dir=str(expdir),
    best_ckpt_callbacks=[valid_loss_ckpt, valid_wer_ckpt],
)
```

This mirrors the behaviour of `get_default_callbacks` while leaving room for
experiments that require a custom checkpoint policy.
