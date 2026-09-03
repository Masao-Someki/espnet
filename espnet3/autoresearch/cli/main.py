"""AutoResearch CLI."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import uuid
from importlib import resources
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.autoresearch.agents.file_agent import FileAgentClient
from espnet3.autoresearch.agents.null_agent import NullAgentClient
from espnet3.autoresearch.agents.python_agent import load_python_agent
from espnet3.autoresearch.applications.hyperparameter_search.config_schema import (
    load_autoresearch_config,
)
from espnet3.autoresearch.core.graph import load_graph
from espnet3.autoresearch.core.scheduler import AutoResearchGraphRuntime
from espnet3.autoresearch.core.serialization import save_yaml
from espnet3.autoresearch.core.state import StateStore
from espnet3.autoresearch.core.tick_jobs import build_tick_loop_script
from espnet3.autoresearch.executors.registry import build_executor
from espnet3.autoresearch.executors.base import JobSpec, Resources
from espnet3.autoresearch.executors.slurm import SlurmConfig, SlurmExecutor
from espnet3.autoresearch.search.noop_search import NoOpSearchProvider
from espnet3.autoresearch.search.web_search import WebSearchProvider


def _build_agent(config, recipe_dir: Path):
    kind = str(config.autoresearch.agent.type)
    if kind == "file":
        return FileAgentClient(
            request_filename=str(
                getattr(config.autoresearch.agent, "request_filename", "agent_request.md")
            ),
            response_filename=str(
                getattr(config.autoresearch.agent, "response_filename", "agent_response.yaml")
            ),
            wait=bool(getattr(config.autoresearch.agent, "wait", False)),
        )
    if kind == "python":
        target = str(getattr(config.autoresearch.agent, "target"))
        return load_python_agent(
            target=target,
            recipe_dir=recipe_dir,
            agent_config=config.autoresearch.agent,
        )
    return NullAgentClient()


def _build_search(config):
    kind = str(getattr(config.autoresearch.search, "type", "noop"))
    if kind == "web":
        return WebSearchProvider(config.autoresearch.search)
    return NoOpSearchProvider()


def _default_graph_path() -> Path:
    resource = resources.files(
        "espnet3.autoresearch.applications.hyperparameter_search"
    ).joinpath("default_graph.yaml")
    with resources.as_file(resource) as path:
        return Path(path)


def _resolve_study_dir(recipe_dir: Path, configured: str | Path) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path.resolve()
    return (recipe_dir / path).resolve()


def _load_runtime(study_dir: Path, recipe_dir: Path):
    config = OmegaConf.load(study_dir / "autoresearch.yaml")
    OmegaConf.resolve(config)
    graph = load_graph(study_dir / "graph.yaml", recipe_dir=recipe_dir)
    state = StateStore(study_dir / "state.sqlite")
    executor_local = build_executor(config.autoresearch.executor.controller)
    executor_trial = build_executor(config.autoresearch.executor.trial)
    trial_resources_default = Resources.from_config(
        getattr(config.autoresearch.executor.trial, "resources", None)
    )
    agent = _build_agent(config, recipe_dir=recipe_dir)
    search = _build_search(config)
    return AutoResearchGraphRuntime(
        study_dir=study_dir,
        recipe_dir=recipe_dir,
        config=config,
        graph=graph,
        state=state,
        executor_local=executor_local,
        executor_trial=executor_trial,
        trial_resources_default=trial_resources_default,
        agent=agent,
        search=search,
    )


def cmd_init(args) -> int:
    recipe_dir = Path(args.recipe_dir).resolve()
    config_path = Path(args.config).resolve()
    config = load_autoresearch_config(recipe_dir, config_path)
    study_dir = _resolve_study_dir(recipe_dir, config.autoresearch.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    graph_src = (
        (recipe_dir / str(config.autoresearch.graph)).resolve()
        if getattr(config.autoresearch, "graph", None)
        else _default_graph_path()
    )
    shutil.copyfile(config_path, study_dir / "autoresearch.yaml")
    shutil.copyfile(graph_src, study_dir / "graph.yaml")
    state = StateStore(study_dir / "state.sqlite")
    state.create_or_load_study(str(config.autoresearch.study_name), study_dir)
    program_path = study_dir / "program.md"
    if not program_path.exists():
        program_path.write_text(
            "# AutoResearch Objective\n\nDescribe the search goal here.\n",
            encoding="utf-8",
        )
    study_id = str(config.autoresearch.study_name)
    if not state.list_node_runs(study_id):
        bootstrap_runtime = _load_runtime(study_dir, recipe_dir)
        state.create_node_run(
            run_id=f"run_{uuid.uuid4().hex}",
            study_id=study_id,
            node_name=bootstrap_runtime.bootstrap_node_name(),
        )
    return 0


def cmd_run(args) -> int:
    runtime = _load_runtime(Path(args.study_dir).resolve(), Path(args.recipe_dir).resolve())
    asyncio.run(runtime.run_forever(poll_interval=float(args.poll_interval)))
    return 0


def cmd_tick(args) -> int:
    study_dir = Path(args.study_dir).resolve()
    recipe_dir = Path(args.recipe_dir).resolve()
    runtime = _load_runtime(study_dir, recipe_dir)
    if (study_dir / "STOP").exists():
        runtime.state.update_study_status(runtime.study_id, "stopped")
        return 0
    runtime.tick(trial_id=getattr(args, "trial_id", None))
    return 0


def cmd_run_node(args) -> int:
    runtime = _load_runtime(Path(args.study_dir).resolve(), Path(args.recipe_dir).resolve())
    if runtime.gpu_self_controller_enabled() and runtime.graph.nodes[args.node].executor == "trial":
        result = runtime.run_gpu_self_controller(
            node_name=args.node,
            run_id=args.run_id,
            trial_id=args.trial_id,
            attempt_id=args.attempt_id,
        )
    else:
        result = runtime.run_node(
            node_name=args.node,
            run_id=args.run_id,
            trial_id=args.trial_id,
            attempt_id=args.attempt_id,
            controller_mode=False,
        )
    return 0 if result.status not in {"failure", "timeout"} else 1


def cmd_status(args) -> int:
    study_dir = Path(args.study_dir).resolve()
    config = OmegaConf.load(study_dir / "autoresearch.yaml")
    state = StateStore(study_dir / "state.sqlite")
    trials = state.list_trials(str(config.autoresearch.study_name))
    print(f"study_dir: {study_dir}")
    print(f"trials: {len(trials)}")
    if trials:
        best = state.get_best_trial(
            str(config.autoresearch.study_name),
            metric=str(config.autoresearch.metric.name),
            mode=str(config.autoresearch.metric.mode),
        )
        if best is not None:
            print(f"best: {best.trial_id} score={best.score}")
    return 0


def cmd_stop(args) -> int:
    study_dir = Path(args.study_dir).resolve()
    (study_dir / "STOP").write_text("stop\n", encoding="utf-8")
    state = StateStore(study_dir / "state.sqlite")
    try:
        config = OmegaConf.load(study_dir / "autoresearch.yaml")
        state.update_study_status(str(config.autoresearch.study_name), "stopped")
    except Exception:  # noqa: BLE001
        pass
    return 0


def cmd_finish_tick_job(args) -> int:
    study_dir = Path(args.study_dir).resolve()
    state = StateStore(study_dir / "state.sqlite")
    state.finish_tick_job(str(args.job_id))
    return 0


def _build_tick_resources(args) -> Resources:
    return Resources(
        gpus=int(getattr(args, "slurm_gpus", 0) or 0),
        cpus=1,
        mem=args.slurm_mem,
        time=args.slurm_time,
        partition=args.slurm_partition,
        account=args.slurm_account,
        qos=args.slurm_qos,
        constraint=args.slurm_constraint,
        reservation=args.slurm_reservation,
        nodelist=args.slurm_nodelist,
        extra=dict(getattr(args, "slurm_extra", {}) or {}),
    )


def _apply_tick_resource_defaults(args, config) -> None:
    tick_resources = getattr(getattr(config.autoresearch.executor, "tick", None), "resources", None)
    # If this is a trial monitor, allow executor.trial_monitor.resources to override tick resources.
    if getattr(args, "trial_id", None):
        monitor_resources = getattr(
            getattr(config.autoresearch.executor, "trial_monitor", None), "resources", None
        )
        if monitor_resources is not None:
            tick_resources = monitor_resources
    if tick_resources is None:
        return
    if getattr(args, "slurm_gpus", None) is None:
        args.slurm_gpus = int(getattr(tick_resources, "gpus", 0) or 0)
    if getattr(args, "slurm_mem", None) in {None, ""}:
        args.slurm_mem = getattr(tick_resources, "mem", None)
    if getattr(args, "slurm_time", None) in {None, ""}:
        args.slurm_time = getattr(tick_resources, "time", None)
    if getattr(args, "slurm_partition", None) in {None, ""}:
        args.slurm_partition = getattr(tick_resources, "partition", None)
    if getattr(args, "slurm_account", None) in {None, ""}:
        args.slurm_account = getattr(tick_resources, "account", None)
    if getattr(args, "slurm_qos", None) in {None, ""}:
        args.slurm_qos = getattr(tick_resources, "qos", None)
    if getattr(args, "slurm_constraint", None) in {None, ""}:
        args.slurm_constraint = getattr(tick_resources, "constraint", None)
    if getattr(args, "slurm_reservation", None) in {None, ""}:
        args.slurm_reservation = getattr(tick_resources, "reservation", None)
    if getattr(args, "slurm_nodelist", None) in {None, ""}:
        args.slurm_nodelist = getattr(tick_resources, "nodelist", None)
    if not getattr(args, "slurm_extra", None):
        args.slurm_extra = dict(getattr(tick_resources, "extra", {}) or {})


def _resolve_tick_interval(args, config, state, study_id: str) -> int:
    active_interval = getattr(config.autoresearch, "tick_interval_sec_trial_active", None)
    idle_interval = getattr(config.autoresearch, "tick_interval_sec_idle", None)
    if getattr(args, "interval", None) not in {None, 0}:
        return int(args.interval)
    if getattr(args, "trial_id", None):
        return int(active_interval if active_interval is not None else 600)
    if idle_interval is not None or active_interval is not None:
        return int(idle_interval if idle_interval is not None else 60)
    configured = getattr(config.autoresearch, "tick_interval_sec", None)
    return int(configured if configured is not None else 300)


def _build_tick_job_spec(
    study_dir: Path,
    recipe_dir: Path,
    interval: int,
    resources: Resources,
    backend: str,
    args,
) -> tuple[JobSpec, Path, Path]:
    trial_id = getattr(args, "trial_id", None)
    tick_kind = "trial_monitor" if trial_id else "tick"
    tick_id = f"{tick_kind}_{uuid.uuid4().hex}"
    job_id = f"ar_{tick_kind}_{uuid.uuid4().hex}"
    tick_dir = study_dir / "ticks" / tick_id
    tick_dir.mkdir(parents=True, exist_ok=True)
    stdout = tick_dir / "stdout.log"
    stderr = tick_dir / "stderr.log"
    submit_stdout = tick_dir / "submit_stdout.log"
    submit_stderr = tick_dir / "submit_stderr.log"
    script_text = build_tick_loop_script(
        study_dir=study_dir,
        recipe_dir=recipe_dir,
        interval=int(interval),
        backend=backend,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        slurm_account=args.slurm_account,
        slurm_mem=args.slurm_mem,
        slurm_qos=args.slurm_qos,
        slurm_constraint=args.slurm_constraint,
        slurm_reservation=args.slurm_reservation,
        slurm_nodelist=args.slurm_nodelist,
        trial_id=str(trial_id) if trial_id else None,
    )
    command = ["bash", "-c", script_text]
    metadata = {
        "job_id": job_id,
        "command": command,
        "resources": resources.__dict__,
        "workdir": str(recipe_dir),
        "script_path": str(tick_dir / "job.sh"),
        "tick_dir": str(tick_dir),
        "submit_stdout_path": str(submit_stdout),
        "submit_stderr_path": str(submit_stderr),
        "job_type": tick_kind,
        "trial_id": trial_id,
    }
    spec = JobSpec(
        name=(
            f"ar-{study_dir.name.replace('_', '-')}-"
            f"{trial_id.replace('_', '-') if trial_id else 'tick'}"
        ),
        command=command,
        workdir=recipe_dir,
        stdout=stdout,
        stderr=stderr,
        resources=resources,
        env={
            "ESPNET_AR_STUDY_DIR": str(study_dir),
            "ESPNET_AR_TICK_JOB_ID": job_id,
        },
        metadata=metadata,
    )
    save_yaml(
        tick_dir / "job.yaml",
        {
            "job_id": job_id,
            "interval": int(interval),
            "backend": backend,
            "command": command,
            "resources": resources.__dict__,
            "job_type": tick_kind,
            "trial_id": trial_id,
            "stdout": str(stdout),
            "stderr": str(stderr),
            "submit_stdout": str(submit_stdout),
            "submit_stderr": str(submit_stderr),
        },
    )
    return spec, stdout, stderr


def cmd_schedule_next_tick(args) -> int:
    study_dir = Path(args.study_dir).resolve()
    recipe_dir = Path(args.recipe_dir).resolve()
    state = StateStore(study_dir / "state.sqlite")
    config = OmegaConf.load(study_dir / "autoresearch.yaml")
    _apply_tick_resource_defaults(args, config)
    study_id = str(config.autoresearch.study_name)
    args.interval = _resolve_tick_interval(args, config, state, study_id)
    study = state.get_study(study_id)
    trial_id = getattr(args, "trial_id", None)
    if (study_dir / "STOP").exists():
        state.update_study_status(study_id, "stopped")
        logging.info("Study is stopped. No next tick scheduled.")
        return 0
    if study.status in {"completed", "failed", "stopped"}:
        logging.info("Study is %s. No next tick scheduled.", study.status)
        return 0
    if trial_id:
        trial = state.get_trial(str(trial_id))
        current_tick_job_id = os.environ.get("ESPNET_AR_TICK_JOB_ID")
        active_trial_jobs = [
            job
            for job in state.list_active_jobs(study_id)
            if job.job_type == "trial" and job.trial_id == str(trial_id)
        ]
        pending_trial_nodes = [
            node_run
            for node_run in state.get_pending_node_runs(study_id)
            if node_run["trial_id"] == str(trial_id)
        ]
        if (
            trial.status not in {"proposed", "running", "completed"}
            and not active_trial_jobs
            and not pending_trial_nodes
        ):
            logging.info("Trial monitor not scheduled because trial %s is inactive.", trial_id)
            return 0
        active_trial_monitors = [
            job
            for job in state.list_active_jobs(study_id)
            if job.job_type == "trial_monitor"
            and job.trial_id == str(trial_id)
            and job.job_id != current_tick_job_id
        ]
        if active_trial_monitors:
            logging.info("Trial monitor already active for %s", trial_id)
            return 0
    backend = str(args.backend)
    if backend == "local":
        logging.warning(
            "schedule-next-tick is designed for Slurm. "
            "For local execution, use: "
            "python -m espnet3.autoresearch.cli.main run --poll-interval %s",
            int(args.interval),
        )
        return 0
    resources = _build_tick_resources(args)
    executor = SlurmExecutor(SlurmConfig())
    spec, stdout, stderr = _build_tick_job_spec(
        study_dir=study_dir,
        recipe_dir=recipe_dir,
        interval=int(args.interval),
        resources=resources,
        backend=backend,
        args=args,
    )
    handle = executor.submit(spec)
    state.create_tick_job(
        study_id=study_id,
        handle=handle,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
    )
    logging.info("Next tick scheduled: slurm_job_id=%s", handle.external_id)
    return 0


def run_from_recipe_config(recipe_dir: Path, config_path: Path, mode: str = "run") -> int:
    """Recipe helper for launching AutoResearch from a recipe config."""
    recipe_dir = Path(recipe_dir).resolve()
    config_path = Path(config_path).resolve()
    init_args = argparse.Namespace(recipe_dir=str(recipe_dir), config=str(config_path))
    cmd_init(init_args)
    config = load_autoresearch_config(recipe_dir, config_path)
    study_dir = _resolve_study_dir(recipe_dir, config.autoresearch.study_dir)
    if mode == "tick":
        return cmd_tick(
            argparse.Namespace(
                study_dir=str(study_dir),
                recipe_dir=str(recipe_dir),
                trial_id=None,
            )
        )
    return cmd_run(
        argparse.Namespace(
            study_dir=str(study_dir),
            recipe_dir=str(recipe_dir),
            poll_interval=30.0,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--recipe-dir", required=True)
    init.add_argument("--config", required=True)
    init.set_defaults(func=cmd_init)

    run = subparsers.add_parser("run")
    run.add_argument("--study-dir", required=True)
    run.add_argument("--recipe-dir", required=True)
    run.add_argument("--poll-interval", type=float, default=30.0)
    run.set_defaults(func=cmd_run)

    tick = subparsers.add_parser("tick")
    tick.add_argument("--study-dir", required=True)
    tick.add_argument("--recipe-dir", required=True)
    tick.add_argument("--trial-id")
    tick.set_defaults(func=cmd_tick)

    finish_tick_job = subparsers.add_parser("finish-tick-job")
    finish_tick_job.add_argument("--study-dir", required=True)
    finish_tick_job.add_argument("--job-id", required=True)
    finish_tick_job.set_defaults(func=cmd_finish_tick_job)

    schedule_next_tick = subparsers.add_parser("schedule-next-tick")
    schedule_next_tick.add_argument("--study-dir", required=True)
    schedule_next_tick.add_argument("--recipe-dir", required=True)
    schedule_next_tick.add_argument("--interval", type=int, required=True)
    schedule_next_tick.add_argument("--trial-id")
    schedule_next_tick.add_argument(
        "--backend",
        choices=["slurm", "local"],
        default="slurm",
    )
    schedule_next_tick.add_argument("--slurm-gpus", type=int, default=None)
    schedule_next_tick.add_argument("--slurm-partition")
    schedule_next_tick.add_argument("--slurm-account")
    schedule_next_tick.add_argument("--slurm-time")
    schedule_next_tick.add_argument("--slurm-mem")
    schedule_next_tick.add_argument("--slurm-qos")
    schedule_next_tick.add_argument("--slurm-constraint")
    schedule_next_tick.add_argument("--slurm-reservation")
    schedule_next_tick.add_argument("--slurm-nodelist")
    schedule_next_tick.set_defaults(func=cmd_schedule_next_tick)

    run_node = subparsers.add_parser("run-node")
    run_node.add_argument("--study-dir", required=True)
    run_node.add_argument("--recipe-dir", required=True)
    run_node.add_argument("--node", required=True)
    run_node.add_argument("--run-id", required=True)
    run_node.add_argument("--trial-id")
    run_node.add_argument("--attempt-id", default="attempt_000000")
    run_node.set_defaults(func=cmd_run_node)

    status = subparsers.add_parser("status")
    status.add_argument("--study-dir", required=True)
    status.set_defaults(func=cmd_status)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--study-dir", required=True)
    stop.set_defaults(func=cmd_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
