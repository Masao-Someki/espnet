# ESPnet3 ASR recipe

## Quick start

```bash
# 0) Edit configs to set paths.
#    Keep `conf/training.yaml:dataset_dir` as the canonical dataset location.
#    When `--training_config` is also passed to `infer` or `measure`, run.py
#    propagates experiment path fields from training into inference/metrics.
#    Standalone inference or metrics configs must define their own `exp_tag`
#    or `exp_dir`.

# 1) Convert LibriSpeech to Hugging Face format (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Train with the default Branchformer configuration
python run.py --stages train --training_config conf/training.yaml

# 3) Decode
python run.py --stages infer --inference_config conf/inference.yaml

# 4) Score
python run.py --stages measure --metrics_config conf/metrics.yaml
```

## AutoResearch (hyperparameter search)

`conf/autoresearch.yaml` configures a minimal, config-driven search loop: an
agent proposes a config patch (dotted keys, restricted to `search_space`),
the patch is applied to `training_config`, `trial.commands` runs (typically
the `train`/`infer`/`measure` stages above, pointed at a per-trial
`exp_dir`), the resulting metric is read back, and the trial is accepted if
it beats the best score seen so far. State lives under
`<study_dir>/trials/<trial_id>/`.

```bash
# Create the study directory (writes autoresearch.yaml and a program.md
# stub there) without running any trials yet.
python -m espnet3.autoresearch init --config conf/autoresearch.yaml

# Edit program.md to describe the objective, then run the search loop.
# Stops at budget.max_trials / budget.max_failures / budget.no_improve_stop,
# or immediately if a STOP file appears under the study directory.
python -m espnet3.autoresearch run --config conf/autoresearch.yaml

# Trial counts by status and the current best score.
python -m espnet3.autoresearch status --config conf/autoresearch.yaml
```

By default (`edit.mode: direct`) an agent may edit any file under the
recipe directory during a trial; set `edit.mode: allowlist` and list glob
patterns under `edit.allowlist` to restrict this -- a trial that edits
outside the allowlist is failed and the offending path(s) are restored.
