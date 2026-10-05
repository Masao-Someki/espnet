"""The trial loop: propose -> sanitize patch -> write configs -> run -> judge -> record.

See the design doc's `loop.py` pseudocode (`init` / `loop` / `should_stop`).
Stage logic ported conceptually (not code) from `espnet3/atlas` on
`origin/espnet3/atlas` (`stages/propose_trial.py`, `run_trial.py`,
`collect_result.py`, `accept_or_reject.py`, `select_best.py`): propose,
apply the patch pre-resolve, run, read the metric, accept/reject against
the running best.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from omegaconf import OmegaConf

from espnet3.autoresearch import study, workspace
from espnet3.autoresearch.agent import AgentRequest, AgentResponseError
from espnet3.autoresearch.config import AutoResearchConfig, render
from espnet3.autoresearch.metrics import MetricNotFoundError, read_metric
from espnet3.autoresearch.patch import apply_dotted_patch, sanitize_patch
from espnet3.autoresearch.runner import run_trial
from espnet3.autoresearch.study import TrialRecord
from espnet3.utils.config_utils import load_and_merge_config

logger = logging.getLogger(__name__)

_DEFAULT_OBJECTIVE_STUB = (
    "# Objective\n\nDescribe what this study should optimize for, then rerun.\n"
)


def resolve_study_dir(config: AutoResearchConfig, recipe_dir: Path) -> Path:
    """Return `config.study_dir`, made absolute under `recipe_dir` if relative."""
    study_dir = Path(config.study_dir)
    return study_dir if study_dir.is_absolute() else recipe_dir / study_dir


def ensure_objective_file(recipe_dir: Path, objective_file: str) -> str:
    """Return `<recipe_dir>/objective_file`'s text, creating a stub if missing."""
    path = recipe_dir / objective_file
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_OBJECTIVE_STUB, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def render_config_yaml(config: AutoResearchConfig) -> str:
    """Render `config` back to YAML text, for the study's `autoresearch.yaml` copy."""
    return OmegaConf.to_yaml(OmegaConf.create(asdict(config)))


def _no_improve_count(records, mode: str) -> int:
    best: Optional[float] = None
    count = 0
    for record in records:
        if record.status == "running" or record.score is None:
            continue
        improved = best is None or (
            record.score < best if mode == "min" else record.score > best
        )
        if improved:
            best = record.score
            count = 0
        else:
            count += 1
    return count


def should_stop(config: AutoResearchConfig, study_dir: Path) -> Tuple[bool, str]:
    """Return `(True, reason)` if the loop should stop before another trial.

    Checked in order: a `STOP` file in `study_dir`, `budget.max_trials`,
    `budget.max_failures`, and `budget.no_improve_stop` consecutive trials
    without a new best score.
    """
    if (study_dir / "STOP").exists():
        return True, "stop_file"
    records = study.load_all_trial_records(study_dir)
    if len(records) >= config.budget.max_trials:
        return True, "max_trials"
    failures = sum(1 for r in records if r.status == "failed")
    if failures >= config.budget.max_failures:
        return True, "max_failures"
    if _no_improve_count(records, config.metric.mode) >= config.budget.no_improve_stop:
        return True, "no_improve_stop"
    return False, ""


def _load_role_config_raw(
    recipe_dir: Path, relative_path: str, config_name: str
) -> Dict[str, Any]:
    cfg = load_and_merge_config(
        recipe_dir / relative_path, config_name=config_name, resolve=False
    )
    return OmegaConf.to_container(cfg, resolve=False)


def _dump_resolved_yaml(raw: Dict[str, Any], path: Path) -> None:
    cfg = OmegaConf.create(raw)
    OmegaConf.resolve(cfg)
    path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")


def _write_trial_configs(
    config: AutoResearchConfig,
    recipe_dir: Path,
    trial_dir: Path,
    patch: Dict[str, Any],
    placeholders: Dict[str, str],
) -> Tuple[Path, Path, Path]:
    """Apply `patch` and `recipe.runtime_keys` (pre-resolve), write 3 YAML files.

    The agent's `patch` is applied only to the training config (inference
    and metrics stay unchanged by the agent, per the design: evaluation is
    not something a trial should be able to alter). `runtime_keys` (trial-dir
    path overrides) are applied to all three, also pre-resolve, so
    interpolations referencing them still resolve correctly.
    """
    training_raw = _load_role_config_raw(
        recipe_dir, config.recipe.training_config, "training.yaml"
    )
    inference_raw = _load_role_config_raw(
        recipe_dir, config.recipe.inference_config, "inference.yaml"
    )
    metrics_raw = _load_role_config_raw(
        recipe_dir, config.recipe.metrics_config, "metrics.yaml"
    )

    training_raw = apply_dotted_patch(training_raw, patch)
    runtime = config.recipe.runtime_keys
    training_raw = apply_dotted_patch(
        training_raw, {k: render(v, placeholders) for k, v in runtime.training.items()}
    )
    inference_raw = apply_dotted_patch(
        inference_raw,
        {k: render(v, placeholders) for k, v in runtime.inference.items()},
    )
    metrics_raw = apply_dotted_patch(
        metrics_raw, {k: render(v, placeholders) for k, v in runtime.metrics.items()}
    )

    training_path = trial_dir / "training.yaml"
    inference_path = trial_dir / "inference.yaml"
    metrics_path = trial_dir / "metrics.yaml"
    _dump_resolved_yaml(training_raw, training_path)
    _dump_resolved_yaml(inference_raw, inference_path)
    _dump_resolved_yaml(metrics_raw, metrics_path)
    return training_path, inference_path, metrics_path


def _recent_failure_log(study_dir: Path, max_chars: int = 2000) -> str:
    for record in reversed(study.load_all_trial_records(study_dir)):
        if record.status != "failed":
            continue
        logs = sorted(study.trial_dir(study_dir, record.trial_id).glob("cmd_*.log"))
        if logs:
            text = logs[-1].read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:]
        return record.reason
    return ""


def _propose(
    agent: Any, request: AgentRequest, *, artifact_dir: Path, placeholders, cwd
):
    """Call `agent.propose`, passing `placeholders`/`cwd` only if it accepts them.

    Keeps `loop.py` decoupled from the concrete `CommandAgent`/`FileAgent`
    classes: any object with a compatible `propose(request, *, artifact_dir,
    ...)` method works.
    """
    kwargs: Dict[str, Any] = {"artifact_dir": artifact_dir}
    params = inspect.signature(agent.propose).parameters
    if "placeholders" in params:
        kwargs["placeholders"] = placeholders
    if "cwd" in params:
        kwargs["cwd"] = cwd
    return agent.propose(request, **kwargs)


def _finish_failed(study_dir: Path, record: TrialRecord, *, reason: str) -> None:
    record.status = "failed"
    record.reason = reason
    record.finished_at = study.utc_now()
    study.write_trial_record(study_dir, record)
    study.append_trials_csv(study_dir, record)


def run_trial_once(
    config: AutoResearchConfig,
    *,
    recipe_dir: Path,
    study_dir: Path,
    agent: Any,
    log: logging.Logger,
) -> TrialRecord:
    """Run exactly one trial (propose -> patch -> run -> read metric -> judge)."""
    trial_id = study.next_trial_id(study_dir)
    trial_dir = study.trial_dir(study_dir, trial_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    record = TrialRecord(
        trial_id=trial_id, status="running", created_at=study.utc_now()
    )
    study.write_trial_record(study_dir, record)
    log.info("=== [START] %s ===", trial_id)

    best = study.read_best(study_dir)
    request = AgentRequest(
        task="propose_config_patch",
        objective=(study_dir / "program.md").read_text(encoding="utf-8"),
        context={
            "edit_mode": config.edit.mode,
            "edit_allowlist": config.edit.allowlist,
            "base_config": config.recipe.training_config,
            "search_space": {
                "allowed_keys": config.search_space.allowed_keys,
                "denied_keys": config.search_space.denied_keys,
            },
            "trial_history_csv": study.read_trials_csv_text(study_dir),
            "best_patch": best["patch"] if best else None,
            "best_score": best["score"] if best else None,
            "recent_failure_log": _recent_failure_log(study_dir),
        },
        allowed_actions=["propose a config_patch (dotted keys) within search_space"],
        output_schema={
            "rationale": "str",
            "config_patch": "dict[str, Any] (dotted keys)",
            "expected_effect": "str",
            "risk": "str",
        },
    )

    placeholders = {
        "recipe_dir": str(recipe_dir),
        "study_dir": str(study_dir),
        "trial_dir": str(trial_dir),
        "trial_id": trial_id,
    }
    try:
        response = _propose(
            agent,
            request,
            artifact_dir=trial_dir / "agent",
            placeholders=placeholders,
            cwd=recipe_dir,
        )
    except AgentResponseError as e:
        _finish_failed(study_dir, record, reason=e.reason)
        log.warning("=== [FAILED] %s: agent_response: %s ===", trial_id, e)
        return record

    patch = sanitize_patch(
        response.config_patch,
        config.search_space.allowed_keys,
        config.search_space.denied_keys,
    )
    record.rationale = response.rationale
    if not patch:
        record.patch = {}
        _finish_failed(study_dir, record, reason="empty_patch")
        log.warning(
            "=== [FAILED] %s: empty_patch (no allowed keys in proposal) ===", trial_id
        )
        return record
    record.patch = patch

    placeholders = {
        **placeholders,
        "exp_dir": str(trial_dir / "exp"),
        "inference_dir": str(trial_dir / "exp" / "inference"),
    }
    training_path, inference_path, metrics_path = _write_trial_configs(
        config, recipe_dir, trial_dir, patch, placeholders
    )
    placeholders = {
        **placeholders,
        "training_config": str(training_path),
        "inference_config": str(inference_path),
        "metrics_config": str(metrics_path),
    }

    try:
        ws_state = workspace.begin(
            config.edit.mode, config.edit.allowlist, recipe_dir, trial_dir
        )
    except workspace.WorkspaceError as e:
        _finish_failed(study_dir, record, reason="workspace_error")
        log.warning("=== [FAILED] %s: workspace_error: %s ===", trial_id, e)
        return record

    result = run_trial(
        commands=config.trial.commands,
        workdir=config.trial.workdir,
        env=config.trial.env,
        timeout_sec=config.trial.timeout_sec,
        placeholders=placeholders,
        trial_dir=trial_dir,
    )

    ws_result = workspace.end(ws_state)
    if ws_result.status == "failed":
        workspace.finalize(ws_state, accepted=False)
        _finish_failed(study_dir, record, reason=ws_result.reason or "edit_violation")
        log.warning(
            "=== [FAILED] %s: %s: paths=%s ===",
            trial_id,
            ws_result.reason,
            ws_result.violation_paths,
        )
        return record

    if result.status != "success":
        workspace.finalize(ws_state, accepted=False)
        _finish_failed(study_dir, record, reason=result.status)
        log.warning(
            "=== [FAILED] %s: %s: %s ===", trial_id, result.status, result.message
        )
        return record

    metric_source = [
        {
            "path": render(entry.path, placeholders),
            "key": entry.key,
            "pattern": entry.pattern,
        }
        for entry in config.metric.source
    ]
    try:
        score = read_metric(metric_source)
    except MetricNotFoundError as e:
        workspace.finalize(ws_state, accepted=False)
        _finish_failed(study_dir, record, reason="metric_not_found")
        log.warning("=== [FAILED] %s: metric_not_found: %s ===", trial_id, e)
        return record

    record.score = score
    best = study.read_best(study_dir)
    accepted = best is None or (
        score < best["score"] if config.metric.mode == "min" else score > best["score"]
    )
    record.status = "accepted" if accepted else "rejected"
    record.finished_at = study.utc_now()
    study.write_trial_record(study_dir, record)
    study.append_trials_csv(study_dir, record)
    if accepted:
        study.update_best(study_dir, record, mode=config.metric.mode)
    study.write_leaderboard(study_dir, mode=config.metric.mode)
    workspace.finalize(ws_state, accepted=accepted)
    log.info("=== [%s] %s score=%s ===", record.status.upper(), trial_id, score)
    return record


def run_study(
    config: AutoResearchConfig,
    *,
    recipe_dir: Path,
    agent: Any,
    log: Optional[logging.Logger] = None,
) -> None:
    """Run (or resume) a study until `should_stop` says to.

    Args:
        config: The validated `autoresearch.yaml`.
        recipe_dir: The recipe directory `trial.commands`/role configs are
            relative to.
        agent: Anything with a `propose(request, *, artifact_dir, ...)`
            method (`agent.CommandAgent`, `agent.FileAgent`, or a test
            double).
        log: Logger to use; defaults to this module's logger.

    Raises:
        AgentWaitingError: Propagated from a `FileAgent` with `wait=False`
            and no response yet -- this pauses the whole run (not just one
            trial), so the caller (`cli.py`) can report "waiting" and exit
            rather than looping forever.
    """
    log = log or logger
    recipe_dir = Path(recipe_dir)
    study_dir = resolve_study_dir(config, recipe_dir)

    objective_text = ensure_objective_file(recipe_dir, config.objective_file)
    study.init_study(
        study_dir,
        config_yaml_text=render_config_yaml(config),
        objective_text=objective_text,
    )
    resumed = study.mark_interrupted_running_trials(study_dir)
    if resumed:
        log.warning(
            "Resuming: marked running trial(s) failed(interrupted): %s", resumed
        )

    while True:
        stop, reason = should_stop(config, study_dir)
        if stop:
            log.info("Stopping study %s: %s", config.study_name, reason)
            return
        run_trial_once(
            config, recipe_dir=recipe_dir, study_dir=study_dir, agent=agent, log=log
        )
