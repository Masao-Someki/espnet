# Configuration

A study is defined by one YAML file, conventionally `conf/atlas.yaml` in the
recipe. It is loaded by
[`load_atlas_config`](https://github.com/espnet/espnet/blob/master/espnet3/bin/atlas/core/config.py),
which merges it over the packaged defaults
(`espnet3/bin/atlas/resources/atlas_defaults.yaml`) and validates
`atlas.submission`. Everything lives under a top-level `atlas:` key.

## Minimal example

```yaml
atlas:
  study_name: mini_an4_ci_check
  study_dir: exp/atlas/mini_an4_ci_check
  application: hyperparameter_search
  objective_file: program.md

  task: ASR
  dataset: mini_an4
  model: transformer
  year: 2026

  recipe:
    training_config: conf/training_asr_transformer.yaml
    inference_config: conf/inference.yaml
    metrics_config: conf/metrics.yaml

  metric:
    name: wer
    mode: min
    extraction:
      - path: "inference/metrics.json"
        name: wer
        key: "espnet3.systems.esp2_asr.metrics.wer.WER.test.WER"

  budget:
    max_trials: 20

  agent:
    provider: claude

  submission:
    resources: {}

  search_space:
    allowed_keys:
      - trainer.*
      - optimizer.*
    denied_keys:
      - data.*
      - output_dir
```

## Core identity

| Key | Meaning |
| --- | --- |
| `study_name` | Unique id for the study; used as the state tables' primary key and the job-name prefix (`ar-<study_name>-...`). |
| `study_dir` | Where the study's frozen config, graph, state, and trials live. Relative paths resolve against `--recipe-dir`. |
| `application` | Name of the graph to run: a packaged graph (`hyperparameter_search`, `architecture_search`) or a path to a graph YAML file in the recipe. See [Graphs and stages](./graphs-and-stages.html). |
| `objective_file` | The objective markdown file (see below), copied into `study_dir/program.md` at `init` time. |
| `task` / `dataset` / `model` / `year` | Free-form metadata recorded with the study; not interpreted by Atlas itself. |

## `recipe`: the configs being searched over

```yaml
atlas:
  recipe:
    training_config: conf/training_asr_transformer.yaml
    inference_config: conf/inference.yaml
    metrics_config: conf/metrics.yaml
```

These are the recipe's own `run.py` configs. Every trial loads
`training_config` unresolved, applies the agent's config patch with dotted
keys (e.g. `optimizer.lr: 3e-4`), resolves it, and writes the result to
`trials/<trial_id>/resolved_training_config.yaml`. `inference_config` and
`metrics_config` are treated as fixed evaluation invariants: only
runtime-generated paths (`inference_dir`, etc.) are substituted into them —
the agent cannot patch evaluation itself.

## `metric`: what a trial is scored on

```yaml
atlas:
  metric:
    name: wer
    mode: min      # or "max"
    extraction:
      - path: "inference/metrics.json"   # relative to the trial dir
        name: wer
        key: "espnet3.systems.esp2_asr.metrics.wer.WER.test.WER"
```

`collect_result` reads `extraction[].path` (relative to the trial directory)
as JSON and pulls out `extraction[].key` (a dotted path into that JSON) as
`extraction[].name`. `metric.name` must match one of those extracted names;
`metric.mode` (`min`/`max`) decides which direction is "better" for
`accept_or_reject` and `select_best`. Multiple `extraction` entries let a
trial record more than one metric even though only one is optimized.

## `budget` and stopping

```yaml
atlas:
  budget:
    max_trials: 20            # hard cap on proposed trials
    max_parallel_trials: 1    # concurrent running trials (default 1)
    max_failures: 10          # stop after this many failed/timeout trials
    no_improve_stop: 10       # stop after this many trials with no new best
```

`check_budget` (and `AtlasGraphRuntime.should_stop`) stops the study as soon
as any of these trigger. All four are optional; a key left unset (or `0`)
disables that particular stopping trigger — only `max_trials` is commonly
required to bound a study.

### In-training early stopping

```yaml
atlas:
  early_stopping:
    enabled: true
    epoch_interval: 1              # check every N epochs (default: ~10 checks per run)
    # iteration_interval: 500      # mutually exclusive with epoch_interval
    # elapsed_time_interval_sec: 1800
```

Independently of the trial budget, ATLAS can ask the coding agent to look at
a *running* trial's training curve at regular keypoints (epoch, iteration, or
elapsed wall-clock time — pick exactly one trigger) and decide whether to
keep training or stop it early as clearly unpromising. This requires GPU
self-controller mode (below), since the check happens from inside the
training job.

## `agent`: which coding-agent CLI drives the study

```yaml
atlas:
  agent:
    provider: claude   # or "codex"
    claude:
      model: null              # use the CLI's default model
      permission_mode: bypassPermissions
      effort: null
    codex:
      model: null
      sandbox: workspace-write
```

`provider` selects the backend (see
`espnet3/bin/atlas/resources/agent_defaults.yaml` for every field, and
`espnet3/bin/atlas/agents/providers/{claude.py,codex.py}` for how each is
invoked). Every call to the agent — proposing a trial, debugging a failure,
reflecting, summarizing — goes through this one configured CLI, run as a
subprocess per `espnet3/bin/atlas/agents/cli.py::CliAgent`.

## `submission`: how trial and controller jobs run

```yaml
atlas:
  submission:
    resources:
      gpus: 1
      cpus: 4
      mem: 32G
      time: "08:00:00"
      extra:
        partition: gpuA100x4
    instructions: |
      A free-form note read by the agent when it needs to understand this
      cluster's job-submission conventions (e.g. which `submit_*.sh` script
      to call, or a raw `sbatch`/`qsub` template).
    controller:
      run_in: local        # "local" (default) or "queue"
      handoff_after_sec: 3600
```

* `resources` is the study-level default `Resources` (gpus/cpus/nodes/mem/time
  plus backend-specific `extra` like Slurm `partition`); a graph node (e.g.
  `run_trial`) can set its own `resources` to override it for that node only.
* `instructions` is required whenever `controller.run_in: queue`; it is how
  the agent (or the submission backend) knows how to actually submit a job
  on this cluster. Leaving `instructions` out and `run_in: local` runs every
  job as a local subprocess — useful for CI and single-machine studies.
* `controller.run_in`:
  * `local` — `espnet3 atlas run` itself becomes the controller loop. Run it
    in `tmux`/`screen` on a login node.
  * `queue` — `espnet3 atlas run` submits the controller as a scheduled
    **tick job** and returns; the tick job re-invokes `atlas run --job-id ...`
    and loops until its walltime is nearly up, then submits its own
    successor (`handoff_after_sec`) so a study with no human attached keeps
    ticking indefinitely. See [Running a study](./workflow.html#the-controller).

### GPU self-controller

```yaml
atlas:
  submission:
    gpu_self_controller:
      enabled: true
```

When enabled, the training job itself performs the periodic keypoint checks
(`early_stopping`) instead of relying on a separate controller tick, letting
a study run entirely inside scheduled GPU jobs with no persistent login-node
or `queue` controller process watching it.

## `search_space`: what the agent is allowed to touch

```yaml
atlas:
  search_space:
    allowed_keys:
      - trainer.*
      - optimizer.*
      - model.peft.*
    denied_keys:
      - data.*
      - output_dir
      - collect_stats_dir
      - recipe_dir
      - exp_dir
      - stats_dir
      - inference_dir
```

Every dotted key in the agent's proposed config patch is checked against
`allowed_keys` (glob patterns; a key must match at least one to be kept) and
then `denied_keys` (glob patterns; a match is always dropped, even if also
allowed) by
[`split_patch_by_search_space`](https://github.com/espnet/espnet/blob/master/espnet3/bin/atlas/core/utils.py).
Keys the runtime injects itself (`recipe_dir`, `data_dir`, `exp_tag`,
`exp_dir`) should always be denied — the system sets them per trial and an
agent patch for them would be silently discarded anyway, but listing them
makes the boundary explicit to whoever reads the config. If the entire patch
is filtered out, the trial is rejected with `status="failure"` rather than
silently proposing nothing.

## `trial`: how a trial is executed

```yaml
atlas:
  trial:
    timeout_sec: 86400          # default: 1 day
    command_template: null      # advanced: override train/infer/measure entirely
```

By default `run_trial` runs `run.py --stages train`, then `infer`, then
`measure` in sequence, each with the trial's resolved configs, resuming from
`exp/last.ckpt` if present and skipping stages a previous attempt already
completed (see `stage_status.yaml` in the trial directory). `command_template`
lets a recipe replace this with an arbitrary list of argv templates
(`{training_config}`, `{inference_config}`, `{metrics_config}`, `{trial_dir}`,
`{attempt_dir}`, `{recipe_dir}` placeholders) for workloads that don't fit the
train/infer/measure shape.

## The objective file (`program.md`)

`atlas.objective_file` (default `program.md`) names a markdown file — next to
`conf/atlas.yaml` or in the recipe root — that is copied into
`study_dir/program.md` by `espnet3 atlas init` and read by *every* agent call
in the study (it is the `objective` field of every `AgentRequest`). Unlike
the YAML keys above, this is free text you write for the agent, not for
Atlas itself. A good `program.md` typically states:

- the concrete objective and the metric/data split it is judged on;
- what is fixed and must not be changed (and *why*, so the agent does not
  "fix" a deliberate constraint — e.g. a `batch_bins` value chosen to avoid
  OOMs on this cluster);
- explicit boundaries ("do not read other recipes under `egs3/`") when the
  coding agent might otherwise wander outside the recipe;
- for a longer budget, an instruction to maintain a written research plan
  (a checklist of hypotheses) so trials spend the budget exploring breadth
  before narrowing in, instead of fixating on the first promising direction.

See `egs3/chime4/asr/program.md` in this repo for a complete, real example.
