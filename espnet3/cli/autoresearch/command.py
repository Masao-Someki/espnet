"""AutoResearch subcommands for ``espnet3``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
import uuid
from importlib import resources
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.cli.autoresearch.agents.cli import CliAgent
from espnet3.cli.autoresearch.applications.hyperparameter_search.config_schema import (
    load_autoresearch_config,
)
from espnet3.cli.autoresearch.core.graph import Resources, load_graph
from espnet3.cli.autoresearch.core.scheduler import AutoResearchGraphRuntime
from espnet3.cli.autoresearch.core.state import StateStore


def _build_agent(config, recipe_dir: Path):
    return CliAgent(config.autoresearch.agent, recipe_dir=recipe_dir)


def _default_graph_path(application: str = "hyperparameter_search") -> Path:
    """Return the packaged default graph for the selected application."""
    if application == "architecture_search":
        package = "espnet3.cli.autoresearch.applications.architecture_search"
        graph_name = "graph.yaml"
    else:
        package = "espnet3.cli.autoresearch.applications.hyperparameter_search"
        graph_name = "default_graph.yaml"
    resource = resources.files(package).joinpath(graph_name)
    with resources.as_file(resource) as path:
        return Path(path)


def _resolve_study_dir(recipe_dir: Path, configured: str | Path) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (recipe_dir / path).resolve()


def _load_runtime(study_dir: Path, recipe_dir: Path):
    config = OmegaConf.load(study_dir / "autoresearch.yaml")
    OmegaConf.resolve(config)
    graph = load_graph(study_dir / "graph.yaml", recipe_dir=recipe_dir)
    state = StateStore(study_dir)
    trial_resources_default = Resources.from_config(
        getattr(config.autoresearch.submission, "resources", None)
    )
    return AutoResearchGraphRuntime(
        study_dir=study_dir,
        recipe_dir=recipe_dir,
        config=config,
        graph=graph,
        state=state,
        trial_resources_default=trial_resources_default,
        agent=_build_agent(config, recipe_dir),
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
        else _default_graph_path(
            str(getattr(config.autoresearch, "application", "hyperparameter_search"))
        )
    )
    shutil.copyfile(config_path, study_dir / "autoresearch.yaml")
    shutil.copyfile(graph_src, study_dir / "graph.yaml")
    state = StateStore(study_dir)
    state.create_or_load_study(str(config.autoresearch.study_name), study_dir)
    program_path = study_dir / "program.md"
    if not program_path.exists():
        objective_file = Path(str(getattr(config.autoresearch, "objective_file", "program.md")))
        sources = (config_path.parent / objective_file, recipe_dir / objective_file)
        source = next((path for path in sources if path.is_file()), None)
        if source is not None:
            shutil.copyfile(source, program_path)
        else:
            program_path.write_text(
                "# AutoResearch Objective\n\nDescribe the search goal here.\n",
                encoding="utf-8",
            )
    study_id = str(config.autoresearch.study_name)
    if not state.list_node_runs(study_id):
        runtime = _load_runtime(study_dir, recipe_dir)
        state.create_node_run(
            run_id=f"run_{uuid.uuid4().hex}",
            study_id=study_id,
            node_name=runtime.bootstrap_node_name(),
        )
    return 0


def _register_tick_job(runtime, resources: Resources):
    study_dir = runtime.study_dir
    recipe_dir = runtime.recipe_dir
    tick_dir = study_dir / "ticks" / f"tick_{uuid.uuid4().hex}"
    tick_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"ar_tick_{uuid.uuid4().hex}"
    stdout, stderr = tick_dir / "stdout.log", tick_dir / "stderr.log"
    command = [
        sys.executable,
        "-m",
        "espnet3.cli.main",
        "autoresearch",
        "run",
        "--study-dir",
        str(study_dir),
        "--recipe-dir",
        str(recipe_dir),
        "--worker",
        "--job-id",
        job_id,
    ]
    metadata = {
        "job_id": job_id,
        "command": command,
        "resources": resources.__dict__,
        "workdir": str(recipe_dir),
        "job_type": "tick",
    }
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join(
        path
        for path in (str(repo_root), os.environ.get("PYTHONPATH", ""))
        if path
    )
    job = runtime.submit_job(
        job_id=job_id,
        job_type="tick",
        name=f"ar-{study_dir.name.replace('_', '-')}-controller",
        command=command,
        workdir=recipe_dir,
        stdout=stdout,
        stderr=stderr,
        resources=resources,
        env={"PYTHONPATH": pythonpath},
        metadata=metadata,
    )
    tick_metadata = {**metadata, "backend": job.backend, "external_id": job.external_id}
    (tick_dir / "job.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(tick_metadata)), encoding="utf-8"
    )
    return job_id


def cmd_run(args) -> int:
    study_dir = Path(args.study_dir).resolve()
    recipe_dir = Path(args.recipe_dir).resolve()
    config = OmegaConf.load(study_dir / "autoresearch.yaml")
    submission_cfg = config.autoresearch.submission
    environment = str(getattr(submission_cfg, "environment", "local"))
    state = StateStore(study_dir)
    study_id = str(config.autoresearch.study_name)
    runtime = _load_runtime(study_dir, recipe_dir)
    controller_cfg = getattr(submission_cfg, "controller", None)
    controller_resources = Resources.from_config(
        getattr(controller_cfg, "resources", None)
    )
    if args.worker:
        job_id = args.job_id or os.environ.get("ESPNET_AR_TICK_JOB_ID")
        handoff_after = int(getattr(controller_cfg, "handoff_after_sec", 0) or 0) or None
        started = time.monotonic()
        try:
            asyncio.run(
                _run_controller_worker(
                    runtime, job_id, controller_resources, handoff_after, started
                )
            )
        finally:
            if job_id:
                state.update_job_status(job_id, "completed")
        return 0
    if environment == "local":
        asyncio.run(runtime.run_forever(poll_interval=args.poll_interval))
        return 0
    active_ticks = [job for job in state.list_active_jobs(study_id) if job.job_type == "tick"]
    if active_ticks:
        logging.info("Controller already active: %s", active_ticks[0].external_id)
        return 0
    _register_tick_job(runtime, controller_resources)
    return 0


async def _run_controller_worker(runtime, job_id, controller_resources, handoff_after, started):
    while True:
        runtime.tick()
        if runtime.should_stop():
            pending = runtime.state.get_pending_node_runs(runtime.study_id)
            active = [
                job for job in runtime.state.list_active_jobs(runtime.study_id)
                if job.job_id != job_id and job.job_type != "tick"
            ]
            if not pending and not active:
                return
        if handoff_after and time.monotonic() - started >= handoff_after:
            active_ticks = [
                job for job in runtime.state.list_active_jobs(runtime.study_id)
                if job.job_type == "tick" and job.job_id != job_id
            ]
            if not active_ticks:
                _register_tick_job(runtime, controller_resources)
            return
        active_trials = runtime.state.list_trials(runtime.study_id, status="running")
        interval_key = "tick_interval_sec_trial_active" if active_trials else "tick_interval_sec_idle"
        interval = int(getattr(runtime.config.autoresearch, interval_key, 600 if active_trials else 60) or 1)
        await asyncio.sleep(max(1, interval))


def cmd_status(args) -> int:
    study_dir = Path(args.study_dir).resolve()
    config = OmegaConf.load(study_dir / "autoresearch.yaml")
    state = StateStore(study_dir)
    study_id = str(config.autoresearch.study_name)
    trials = state.list_trials(study_id)
    print(f"study_dir: {study_dir}")
    print(f"trials: {len(trials)}")
    if trials:
        best = state.get_best_trial(
            study_id,
            metric=str(config.autoresearch.metric.name),
            mode=str(config.autoresearch.metric.mode),
        )
        if best is not None:
            print(f"best: {best.trial_id} score={best.score}")
    return 0


def cmd_stop(args) -> int:
    study_dir = Path(args.study_dir).resolve()
    (study_dir / "STOP").write_text("stop\n", encoding="utf-8")
    try:
        config = OmegaConf.load(study_dir / "autoresearch.yaml")
        StateStore(study_dir).update_study_status(
            str(config.autoresearch.study_name), "stopped"
        )
    except Exception:  # noqa: BLE001
        pass
    return 0


def run_from_recipe_config(recipe_dir: Path, config_path: Path, mode: str = "run") -> int:
    """Recipe helper for launching AutoResearch from a recipe config."""
    recipe_dir, config_path = Path(recipe_dir).resolve(), Path(config_path).resolve()
    cmd_init(argparse.Namespace(recipe_dir=str(recipe_dir), config=str(config_path)))
    config = load_autoresearch_config(recipe_dir, config_path)
    study_dir = _resolve_study_dir(recipe_dir, config.autoresearch.study_dir)
    return cmd_run(argparse.Namespace(
        study_dir=str(study_dir), recipe_dir=str(recipe_dir), poll_interval=None, worker=False
    ))


def add_arguments(subparsers) -> None:
    parser = subparsers.add_parser(
        "autoresearch", help="Run and manage AutoResearch studies.",
        description="Run and manage ESPnet3 AutoResearch studies.",
    )
    commands = parser.add_subparsers(dest="autoresearch_command", required=True)
    init = commands.add_parser("init", help="Initialize a study from a recipe config.")
    init.add_argument("--recipe-dir", required=True)
    init.add_argument("--config", required=True)
    init.set_defaults(func=cmd_init)
    run = commands.add_parser("run", help="Run the study using its configured submission environment.")
    run.add_argument("--study-dir", required=True)
    run.add_argument("--recipe-dir", required=True)
    run.add_argument("--poll-interval", type=float, default=None)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--job-id", default=None, help=argparse.SUPPRESS)
    run.set_defaults(func=cmd_run)
    status = commands.add_parser("status", help="Show study status and best trial.")
    status.add_argument("--study-dir", required=True)
    status.set_defaults(func=cmd_status)
    stop = commands.add_parser("stop", help="Stop a running study.")
    stop.add_argument("--study-dir", required=True)
    stop.set_defaults(func=cmd_stop)
