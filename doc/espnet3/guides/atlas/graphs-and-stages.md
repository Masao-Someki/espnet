# Graphs and stages

A study is driven by a **graph**: a small state machine of named **nodes**,
loaded from `graph.yaml` by
[`load_graph`](https://github.com/espnet/espnet/blob/master/espnet3/bin/atlas/core/graph.py)
into a `ResearchGraph` of `NodeConfig`s
(`espnet3/bin/atlas/core/models.py`). Every tick, the controller
(`AtlasGraphRuntime`, `espnet3/bin/atlas/core/scheduler.py`) finds the next
pending node run, executes its `target` stage, and routes to the next node
by looking up the stage's result `status` in that node's `on:` mapping.

```text
node.execute() -> StageResult(status=...) -> graph.route(node, status) -> next node
```

## Node anatomy

```yaml
run_trial:
  target: espnet3.bin.atlas.stages.run_trial.RunTrialStage
  execution: submission       # "local" or "submission"
  resources:                  # only meaningful for execution: submission
    gpus: 1
    cpus: 4
    mem: 32G
    time: "08:00:00"
  "on":
    success: collect_result
    failure: debug_trial
    timeout: debug_trial
    waiting_job: run_trial
    early_stopped: reflect
```

- `target` — dotted `module.Class` path of a `BaseStage` subclass (see
  below) that implements the node's logic.
- `execution`:
  - `local` — runs synchronously inside the controller tick.
  - `submission` — dispatched as a separate job (cluster or local process,
    per `atlas.submission`), using this node's `resources` merged over the
    study-level default (node values win; see `Resources.merge_with`).
- `on` — maps every `StageResult.status` this stage can return to the name
  of the next node to run. A status with no entry raises an error, so a
  node's `on:` map must cover every status its stage can produce (see each
  stage's docstring below for its status set). `terminal: true` instead of
  an `on:` map ends the graph.
- `skill`, `allowed_actions`, `output_schema`, `output_file`, `output_key`,
  `output_append`, `fallback_output`, `trial_update_keys`,
  `skill_context` — only used by nodes whose `target` is the generic
  `SkillStage` (see [Skill nodes](#skill-nodes-and-skills) below).

`BaseStage.execute` (the common wrapper every stage runs through) caches a
node run's result as a success/failure marker file, so re-dispatching the
same node run (after a crash or restart) returns the cached result instead
of re-running — this is what makes `espnet3 atlas run` safe to just run
again after an interruption.

## The two packaged graphs

Both ship under `espnet3/bin/atlas/graphs/` and are selected by
`atlas.application`.

### `hyperparameter_search`

Pure config-patch search — a trial can only change values in the training
config, never recipe code:

```text
research_context ──► propose_trial ──► run_trial ──► collect_result ──► accept_or_reject
                           │(implement_code)  │(failure/timeout)             │
                           ▼                  ▼                              ▼
                     implement_code ──►   debug_trial ──► (run_trial/reflect)  reflect ──► update_knowledge ──► check_budget
                                                                                                                      │
                                                                      (continue) propose_trial ◄───┘  (done) select_best ──► analyze [terminal]
```

(`implement_code` here is only reached if a node's `skill_context`/search
space still allows writing a new transform under `src/transforms`, per
`program.md`; most hyperparameter-search studies never take that edge.)

### `architecture_search`

Same overall shape, but a proposal can request actual code changes, which
run through a write/test loop before training starts:

```text
... propose_trial ──(implement_code)──► implement_code (WriteFilesStage)
                                               │success        │failure
                                               ▼                ▼
                                          run_tests ──────► run_trial
                                      (failure loops back to implement_code,
                                       success or max_retries_exceeded -> run_trial)
```

`implement_code` here targets `WriteFilesStage` (apply a code patch the
agent already produced) feeding into `RunTestsStage` (run the recipe's
tests against that patch) before a trial is allowed to train — catching a
broken change before it burns a GPU allocation.

Both graphs share the same tail: `collect_result` → `accept_or_reject` →
`reflect` → `update_knowledge` → `check_budget`, looping back to
`propose_trial` until the budget is spent, then `select_best` → `analyze`
(terminal).

## Built-in stages

All under `espnet3/bin/atlas/stages/`, one class per node type:

| Stage | Node(s) | What it does |
| --- | --- | --- |
| `SkillStage` | `research_context`, `reflect`, `update_knowledge`, `analyze`, and any custom skill node | Generic: builds an `AgentRequest` from the node's `skill`, dispatches it to the coding agent, applies `trial_update_keys` to the current trial, writes `output_file`. See below. |
| `ProposeTrialStage` | `propose_trial` | Asks the agent for a config patch (and optionally extra stages / code changes), filters it through `search_space`, deduplicates against prior/in-flight trials (both the raw patch and the *effective resolved config*), writes the trial's resolved configs. |
| `WriteFilesStage` | `implement_code` (architecture search) | Applies a previously-agreed code patch to the recipe's `src/` tree. |
| `RunTestsStage` | `run_tests` (architecture search) | Runs the recipe's test suite against the patched code; routes back to `implement_code` on failure (with the failure fed into the next prompt) up to a retry cap. |
| `RunTrialStage` | `run_trial` | Runs `train`/`infer`/`measure` (or a custom `command_template`) as a local subprocess pipeline; resumes from checkpoint, skips already-completed sub-stages, optionally runs training under GPU self-controller early-stop monitoring. See [Configuration](./configuration.html#trial-how-a-trial-is-executed). |
| `DebugTrialStage` | `debug_trial` | On trial failure/timeout, asks the agent to diagnose and patch/retry (`success` → retry `run_trial`) or give up (`continue`/`failure` → `reflect`). |
| `CollectResultStage` | `collect_result` | Extracts `atlas.metric.extraction` keys from the trial's metrics JSON into `trial.score`/`trial.metrics`. |
| `AcceptOrRejectStage` | `accept_or_reject` | Compares the trial's score with the current best (per `metric.mode`); accepts (new best) or rejects, writes `decision.md`. |
| `CheckBudgetStage` | `check_budget` | Asks the scheduler's `should_stop()` (budget/early-stop rules); routes `continue` or `done`. |
| `SelectBestStage` | `select_best` | Picks the final best trial once the budget is spent. |
| `StageStatus` helpers | used internally by `RunTrialStage` | Tracks per-attempt `train`/`infer`/`measure` completion in `trial_dir/stage_status.yaml` for resumable retries. |

## Skill nodes and skills

Most of a study's "intelligence" lives not in Python but in markdown **skill**
files under `.agents/skills/atlas_<application>_<skill_name>/SKILL.md`,
loaded through `SkillStage`. A skill node's config just names which skill to
run and what shape of reply to require:

```yaml
reflect:
  target: espnet3.bin.atlas.stages.skill.SkillStage
  execution: local
  skill: atlas_hyperparameter_search_reflect
  allowed_actions: [read_only, terminal, file_write]
  trial_update_keys: [reflection, next_suggestion]
  output_file: trials/{trial_id}/reflection.md
  output_key: reflection
  output_schema:
    reflection: {type: string, description: "summary of the trial result"}
    next_suggestion: {type: string, description: "recommended next experiment"}
  "on":
    continue: update_knowledge
    done: update_knowledge
    failure: update_knowledge
```

`SkillStage` builds an `AgentRequest` carrying the study objective
(`program.md`), the current knowledge pack, `trials.csv` history, the
current trial (if any) serialized to JSON, and `skill_context`; it
constrains the agent's reply to a JSON Schema built from `output_schema`
plus a mandatory `status`/`message`, so the agent's answer is always
machine-routable. If `output_key`/`output_file` are set, that field of the
reply is written (or appended, with `output_append: true`) to the named
file under the study directory — this is how `knowledge_pack.md`,
`reflection.md`, and `summary.md` get produced. `trial_update_keys` copies
named reply fields straight onto the in-flight `Trial` record (e.g.
`reflection`/`next_suggestion`) — only fields that already exist on `Trial`
are accepted. `allowed_actions` restricts what the agent is allowed to do
for this call (e.g. `read_only` vs. allowing `file_write`/`terminal`); if
the call fails and the node has a `fallback_output`, that text is written
instead of failing the whole node (used for `research_context`/`analyze` so
a flaky agent call doesn't stall the whole graph).

The shipped skills (`.agents/skills/atlas_hyperparameter_search_*`,
`atlas_study_status`) are:

| Skill | Node | Purpose |
| --- | --- | --- |
| `research_context` | `research_context` | Write the study's first knowledge-pack entry from `program.md` and the repo. |
| `propose_trial` | (used inside `ProposeTrialStage`, not a bare skill node) | Propose the next trial's config patch (and, for architecture search, code changes). |
| `implement_code` | `implement_code` | Write/modify `src/` code for an architecture-search trial. |
| `debug_trial` | (used inside `DebugTrialStage`) | Diagnose a failed/timed-out trial and decide whether to patch-and-retry or give up. |
| `reflect` | `reflect` | Summarize what a completed trial showed and suggest the next direction. |
| `update_knowledge` | `update_knowledge` | Append the trial's takeaway to the cumulative knowledge pack. |
| `analyze` | `analyze` | Write the study's final summary once the budget is spent. |
| `atlas_study_status` | (consulted via `espnet3 atlas status`, not a graph node) | How to read a study's progress without touching it. |

## Writing a custom graph

A recipe can point `atlas.application` at its own graph YAML instead of a
packaged one (resolved relative to `--recipe-dir` first, falling back to the
packaged graphs by name). The two shipped graphs above are a reasonable
starting template: keep the `research_context` → `propose_trial` →
`run_trial` → `collect_result` → `accept_or_reject` → `reflect` →
`update_knowledge` → `check_budget` backbone, and only change which nodes
feed into `propose_trial`/`run_trial` if your trial workload needs an
extra step (e.g. a `train_tokenizer` or `create_dataset` stage first — see
`Trial.extra_stages` and `RunTrialStage._build_default_commands`).
