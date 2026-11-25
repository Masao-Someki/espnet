# ATLAS

ATLAS is ESPnet3's agent-driven research loop. Instead of you hand-writing a
sweep and babysitting it, ATLAS drives a coding agent (Claude Code or Codex)
through a repeating cycle: read the objective and what has been learned so
far, propose a change, run it, judge the result, write down what was learned,
and decide whether to keep going. It builds on the same `run.py` recipe
pipeline every ESPnet3 recipe already uses — ATLAS does not replace
`train`/`infer`/`measure`, it drives them.

A single ATLAS run over a recipe is called a **study**. A study proposes and
evaluates a sequence of **trials** (one trial = one concrete set of config
changes, trained and measured end to end) until a trial budget is spent or a
stopping condition is hit, then writes a final summary of what worked.

::: warning ATLAS was formerly "autoresearch"
Earlier branches and some recipe config filenames (`conf/autoresearch.yaml`,
job-name prefixes like `ar-`) still use the old name. The code now lives
under `espnet3/bin/atlas/` and is invoked as `espnet3 atlas ...`; the naming
in older recipes is cosmetic and does not affect behavior.
:::

## Try it

This is a simulation that runs entirely in your browser: no real agent or
GPU job is launched, and every trial, log line, and file shown is a sample.
Click through `atlas init` → edit `program.md` → `atlas run` to see how a
study's graph, trials, and knowledge pack evolve tick by tick.

<AtlasDemoV2 />

## Why a loop instead of a sweep

A grid or Bayesian sweep explores a config space mechanically; it cannot read
your training logs, notice that a run OOM'd for a reason unrelated to the
hyperparameter, decide to patch the code instead of just tuning a knob, or
write a plan in prose and follow it. ATLAS hands those judgment calls to a
coding agent instead:

- it reads `program.md` (the objective you wrote) and the running
  **knowledge pack** (what every previous trial learned) before proposing
  anything;
- a trial is not limited to a config patch — a graph node can let the agent
  edit recipe code (new transform, new loss term) when the search space
  requires it;
- when a trial fails or times out, a dedicated debug node gets a chance to
  diagnose and retry before the trial is written off;
- after every trial it writes a reflection and appends to the knowledge pack,
  so trial *N+1*'s prompt is grounded in everything trial *1..N* found.

## How it fits together

<DocCards :cols="2">
  <DocCard
    title="Configuration"
    desc="The atlas.yaml study config: objective, metric, budget, search space, submission."
    icon="tabler:settings-2"
    href="./configuration.html"
  />
  <DocCard
    title="Running a study"
    desc="espnet3 atlas init/run/status/stop, the study directory layout, and local vs. queue controllers."
    icon="tabler:player-play"
    href="./workflow.html"
  />
  <DocCard
    title="Graphs and stages"
    desc="The node graph that drives a study, the built-in stages, and the coding-agent skills behind them."
    icon="tabler:topology-star-3"
    href="./graphs-and-stages.html"
  />
</DocCards>

## A study at a glance

```text
<recipe_dir>/
  run.py
  conf/
    atlas.yaml              # study config (this guide's "Configuration" page)
    program.md              # the objective, written by you
    training_asr_*.yaml     # base training/inference/metrics config the search patches
  exp/atlas/<study_name>/    # the study directory (atlas.study_dir)
    atlas.yaml               # frozen copy of the resolved study config
    graph.yaml               # frozen copy of the graph (atlas.application)
    program.md               # frozen copy of the objective
    knowledge/
      knowledge_pack.md       # cumulative learnings, appended after every trial
    trials/
      trial_000001/
        resolved_training_config.yaml
        train.log  eval.log
        metrics.json
        reflection.md
        agent_calls/
      trial_000002/ ...
    node_runs/                # per-graph-node-run artifacts (agent prompts/outputs)
    state_*.csv                # studies / node_runs / trials / jobs / events tables
    .state.lock
    STOP                       # created by `espnet3 atlas stop`
```

`espnet3 atlas init` creates everything above the `trials/`/`node_runs/`
directories; `espnet3 atlas run` drives the graph and fills them in.

## Minimal example

```bash
espnet3 atlas init --recipe-dir egs3/mini_an4/esp2_asr --config conf/atlas.yaml
espnet3 atlas run  --recipe-dir egs3/mini_an4/esp2_asr \
                    --study-dir  egs3/mini_an4/esp2_asr/exp/atlas/mini_an4_ci_check
espnet3 atlas status --study-dir egs3/mini_an4/esp2_asr/exp/atlas/mini_an4_ci_check
```

See [Running a study](./workflow.html) for what each command does and how to
run a study unattended on a cluster.
