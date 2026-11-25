# Running a study

ATLAS is driven entirely through the `espnet3 atlas` subcommand
(`espnet3/bin/atlas/command.py`): `init`, `run`, `status`, `stop`. All four
take `--recipe-dir` and/or `--study-dir`; nothing else needs to be running.

## 1. `init`: create the study

```bash
espnet3 atlas init --recipe-dir egs3/chime4/asr --config conf/atlas.yaml
```

This:

1. creates `atlas.study_dir` (resolved relative to `--recipe-dir` if not
   absolute) and its `knowledge/`, `trials/`, `node_runs/` subdirectories;
2. resolves `conf/atlas.yaml` over the Atlas defaults and freezes it as
   `study_dir/atlas.yaml` — later commands read this frozen copy, not the
   recipe's original config, so editing `conf/atlas.yaml` after `init` has no
   effect on an already-initialized study;
3. copies the graph named by `atlas.application` (a packaged graph, or a
   path in the recipe) to `study_dir/graph.yaml`;
4. copies the objective file to `study_dir/program.md` — or writes a one-line
   template if none exists yet;
5. registers the study and queues its first (bootstrap) node run in state.

Running `init` again on an already-initialized study only refreshes the
frozen config/graph/program files; it does **not** reset trial or node-run
state. This makes it safe to re-run `init` after fixing a typo in
`program.md`.

## 2. `run`: drive the graph

```bash
espnet3 atlas run --recipe-dir egs3/chime4/asr \
                   --study-dir  egs3/chime4/asr/exp/atlas/my_study
```

This starts the **controller**: the loop that executes local graph nodes,
submits trial jobs, and notices when a submitted job has died. Where it
actually runs depends on `atlas.submission.controller.run_in`
(see [Configuration](./configuration.html#submission-how-trial-and-controller-jobs-run)):

### `run_in: local` (default)

The `atlas run` process itself is the controller. It loops, ticking the
graph, until the study finishes or you kill it (or write a `STOP` file).
Since this process must stay alive for the whole study, run it in `tmux` or
`screen` on a login node:

```bash
tmux new -s atlas-my-study
espnet3 atlas run --recipe-dir . --study-dir exp/atlas/my_study
# detach with ctrl-b d; `espnet3 atlas status` to check on it later
```

### `run_in: queue`

```bash
espnet3 atlas run --recipe-dir . --study-dir exp/atlas/my_study
```

submits the controller itself as a scheduled job (a **tick job**) using
`atlas.submission.instructions`/`resources.controller.resources`, and
returns immediately — nothing needs to stay alive on the login node. The
tick job runs the exact same command with an internal `--job-id`, loops
until close to its walltime, then submits its own successor before exiting
("handoff"), so the study keeps ticking indefinitely with no single job
needing a long walltime:

```text
time ───────────────────────────────►
trial jobs  [─ trial 1 ─][─ trial 3 ─] ...   (independent, submitted by the controller)
tick job A  [tick·tick·tick]··✕              ✕ = walltime approaching
                           ▲ handoff: A submits B, exits
tick job B                   [tick·tick·tick]··✕
                                            ▲ handoff: B submits C, ...
```

Calling `espnet3 atlas run` again while a tick job is already alive is a
no-op (it reconciles and finds the existing tick job rather than submitting
a duplicate), so it is safe to re-run after, say, a login-node reboot.

Each tick:

1. runs every local graph node that's ready (proposing a trial, judging a
   result, reflecting, updating knowledge, etc. — see
   [Graphs and stages](./graphs-and-stages.html));
2. submits any node marked `execution: submission` (training/eval) as a
   cluster or local-process job;
3. reconciles job status, so a job that died outside Atlas's notice (killed,
   OOM-killed, node failure) is detected and routed to failure handling
   instead of hanging the study forever;
4. sleeps for `atlas.tick_interval_sec_trial_active` (default 600s) while a
   trial is running, or `atlas.tick_interval_sec_idle` (default 60s)
   otherwise, then repeats.

## 3. `status`: check progress without touching anything

```bash
espnet3 atlas status --study-dir egs3/chime4/asr/exp/atlas/my_study
```

```text
study_dir: egs3/chime4/asr/exp/atlas/my_study
trials: 7
best: trial_000004 score=0.082
trial_status: accepted=2 failed=1 rejected=3 running=1
stop_file: no
active_jobs: 1
  trial_000007 trial running id=3046983.pbs1
```

This reads only the recorded state tables and trial logs — it never queries
the job scheduler directly, so a job's shown status can lag slightly behind
reality (corrected on the controller's next tick). It is always safe to run
from anywhere, concurrently with a running study.

## 4. `stop`: wind a study down gracefully

```bash
espnet3 atlas stop --study-dir egs3/chime4/asr/exp/atlas/my_study
```

Writes a `STOP` file inside the study directory. The controller (and any
tick job) checks for this file every iteration and exits as soon as it sees
it — in-flight trial jobs are left to finish, but no new tick/controller
work is submitted. This is the safe way to halt a study; killing the
controller process directly can leave a submitted tick job running
unsupervised until it hits `handoff_after_sec`.

## Resuming after an interruption

Every stage caches its result under `study_dir/node_runs/<run_id>/` as a
success/failure marker (`BaseStage.execute`), and `run_trial` records which
`train`/`infer`/`measure` sub-stage of a trial has completed
(`trial_dir/stage_status.yaml`). So re-running `espnet3 atlas run` after a
crash, a preemption, or a login-node reboot does not redo finished work: it
picks up the first pending node run or incomplete trial stage and continues
from there. A trial resumes training from `exp/last.ckpt` automatically when
one exists.

## Inspecting what the agent actually did

Every coding-agent call is kept on disk, never only in memory:

```text
<artifact_dir>/agent_calls/call_<uuid>/
  agent_prompt.txt     # exact prompt sent to the CLI
  agent_schema.json    # JSON Schema the reply was constrained to (if any)
  agent_output.txt     # the parsed final reply
  agent_stdout.log     # raw CLI stdout
  agent_stderr.log     # raw CLI stderr
  agent_parse_error.json   # only present if the reply failed schema validation
```

`<artifact_dir>` is a trial directory (`trials/<trial_id>/`) for trial-level
skills (propose, debug, reflect) or a node-run directory
(`node_runs/<run_id>/`) for study-level skills (`research_context`,
`analyze`). This is the first place to look when a study's behavior looks
wrong — it shows exactly what the agent was told and what it answered.
