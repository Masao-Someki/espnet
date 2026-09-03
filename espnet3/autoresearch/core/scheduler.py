"""Graph runtime and scheduler."""

from __future__ import annotations

import asyncio
import csv
import json
import os
from datetime import datetime, timezone
import logging
import shlex
import sys
import uuid
from pathlib import Path

from espnet3.autoresearch.core.context import StageContext
from espnet3.autoresearch.core.errors import ConfigValidationError
from espnet3.autoresearch.core.loader import load_stage_class
from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.serialization import dumps_json, loads_json, named_lock, utc_now
from espnet3.autoresearch.core.state import JobRecord
from espnet3.autoresearch.core.tick_jobs import build_tick_loop_script
from espnet3.autoresearch.executors.base import JobSpec, Resources
from espnet3.autoresearch.executors.slurm import SlurmConfig, SlurmExecutor
from espnet3.autoresearch.agents.session import StudySessionAgentClient


class AutoResearchGraphRuntime:
    """Graph-based AutoResearch runtime."""

    def __init__(
        self,
        study_dir: Path,
        recipe_dir: Path,
        config,
        graph,
        state,
        executor_local,
        executor_trial,
        trial_resources_default,
        agent,
        search,
    ) -> None:
        self.study_dir = Path(study_dir).resolve()
        self.recipe_dir = Path(recipe_dir).resolve()
        self.config = config
        self.graph = graph
        self.state = state
        self.executor_local = executor_local
        self.executor_trial = executor_trial
        self.trial_resources_default = trial_resources_default
        self.search = search
        self.study_id = str(config.autoresearch.study_name)
        self.agent = StudySessionAgentClient(
            agent,
            state,
            self.study_id,
            self.study_dir,
            configured_session_id=getattr(config.autoresearch.agent, "session_id", None),
        )
        self.logger = logging.getLogger("espnet3.autoresearch")
        self._validate_early_stopping_config()

    def _validate_early_stopping_config(self) -> None:
        """Require early stopping to use at most one progress clock."""
        early_stop_cfg = getattr(self.config.autoresearch, "early_stopping", None)
        if early_stop_cfg is None:
            return

        epoch_value = getattr(early_stop_cfg, "epoch_interval", None)
        try:
            epoch_enabled = epoch_value is None or int(epoch_value) > 0
            iteration_enabled = int(
                getattr(early_stop_cfg, "iteration_interval", 0) or 0
            ) > 0
            elapsed_enabled = int(
                getattr(early_stop_cfg, "elapsed_time_interval_sec", 0) or 0
            ) > 0
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                "early_stopping intervals must be null or integer seconds/iterations."
            ) from exc

        enabled = [
            name
            for name, is_enabled in (
                ("epoch_interval", epoch_enabled),
                ("iteration_interval", iteration_enabled),
                ("elapsed_time_interval_sec", elapsed_enabled),
            )
            if is_enabled
        ]
        if len(enabled) > 1:
            raise ConfigValidationError(
                "early_stopping accepts only one trigger, but multiple are enabled: "
                f"{', '.join(enabled)}. Set unused intervals to 0. "
                "In particular, set epoch_interval: 0 when using iteration_interval "
                "or elapsed_time_interval_sec."
            )

    def gpu_self_controller_enabled(self) -> bool:
        """Whether trial workers advance local graph nodes themselves."""
        value = getattr(self.config.autoresearch, "gpu_self_controller", False)
        if isinstance(value, bool):
            return value
        return bool(getattr(value, "enabled", False))

    def gpu_self_monitor_interval_sec(self) -> int:
        value = getattr(self.config.autoresearch, "gpu_self_controller", None)
        configured = getattr(value, "monitor_interval_sec", None)
        if configured is not None:
            return max(1, int(configured))
        return max(
            1,
            int(getattr(self.config.autoresearch, "tick_interval_sec_trial_active", 600) or 600),
        )

    def bootstrap_node_name(self) -> str:
        knowledge_dir = self.study_dir / "knowledge"
        knowledge_ready = all(
            (knowledge_dir / name).is_file()
            for name in ("web_research.md", "repo_context.md", "knowledge_pack.md")
        )
        if knowledge_ready and "propose_trial" in self.graph.nodes:
            return "propose_trial"
        study_ready = all(
            (self.study_dir / name).exists()
            for name in ("autoresearch.yaml", "graph.yaml", "state.sqlite")
        )
        if study_ready and "research_context" in self.graph.nodes:
            return "research_context"
        return self.graph.start

    def _job_name(self, node_name: str, trial_id: str | None, attempt_id: str) -> str:
        study = self.study_dir.name.replace("_", "-")
        if trial_id:
            trial = trial_id
            if trial.startswith("trial_"):
                suffix = trial.split("_", 1)[1]
                if suffix.isdigit():
                    trial = f"trial-{int(suffix):04d}"
                else:
                    trial = trial.replace("_", "-")
            else:
                trial = trial.replace("_", "-")
            return f"ar-{study}-{trial}"[:200]
        node = node_name.replace("_", "-")
        return f"ar-{study}-{node}"[:200]

    def run_node(
        self,
        node_name: str,
        run_id: str,
        trial_id: str | None = None,
        attempt_id: str | None = None,
        controller_mode: bool = False,
    ) -> StageResult:
        node_cfg = self.graph.nodes[node_name]
        stage_cls = load_stage_class(node_cfg.target, recipe_dir=self.recipe_dir)
        stage = stage_cls()
        trial = self.state.get_trial(trial_id) if trial_id else None
        if attempt_id is None:
            attempt_id = "attempt_000000"
        context = StageContext(
            study_dir=self.study_dir,
            recipe_dir=self.recipe_dir,
            node_name=node_name,
            run_id=run_id,
            attempt_id=attempt_id,
            graph=self.graph,
            config=self.config,
            state=self.state,
            executor=self.executor_local if node_cfg.executor == "local" else self.executor_trial,
            agent=self.agent,
            search=self.search,
            logger=self.logger,
            scheduler=self,
            current_trial=trial,
        )
        self.state.begin_node_run(run_id)
        self.state.append_event(
            self.study_id,
            "node_started",
            {"run_id": run_id, "node_name": node_name, "trial_id": trial_id},
        )
        result = stage.execute(context)
        self.state.finish_node_run(run_id, result)
        self.state.append_event(
            self.study_id,
            "node_finished",
            {
                "run_id": run_id,
                "node_name": node_name,
                "trial_id": trial_id,
                "status": result.status,
                "message": result.message,
            },
        )
        self._schedule_next(
            current_node=node_name,
            result=result,
            current_trial_id=trial_id,
            controller_mode=controller_mode,
        )
        return result

    def run_gpu_self_controller(
        self,
        node_name: str,
        run_id: str,
        trial_id: str | None = None,
        attempt_id: str | None = None,
    ) -> StageResult:
        """Finish a trial and advance its local graph nodes inside the GPU job."""
        result = self.run_node(
            node_name=node_name,
            run_id=run_id,
            trial_id=trial_id,
            attempt_id=attempt_id,
            controller_mode=True,
        )

        # A normal controller would observe this job through Slurm.  In this
        # mode the worker is the controller, so it must release its own slot
        # before selecting and submitting the next trial.
        job = self.state.find_job_by_node_run(run_id)
        if job and job.status in {"submitted", "pending", "running", "unknown"}:
            terminal_status = (
                "failed" if result.status in {"failure", "timeout"} else "completed"
            )
            self.state.update_job_status(
                job.job_id, terminal_status, external_id=job.external_id
            )
            self.state.append_event(
                self.study_id,
                "gpu_self_controller_job_finished",
                {
                    "job_id": job.job_id,
                    "run_id": run_id,
                    "status": terminal_status,
                },
            )
        self._submit_pending_trial_nodes()
        return result

    def _build_run_node_command(
        self,
        node_name: str,
        run_id: str,
        trial_id: str | None,
        attempt_id: str,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "espnet3.autoresearch.cli.main",
            "run-node",
            "--study-dir",
            str(self.study_dir),
            "--recipe-dir",
            str(self.recipe_dir),
            "--node",
            node_name,
            "--run-id",
            run_id,
            "--attempt-id",
            attempt_id,
        ]
        if trial_id:
            command.extend(["--trial-id", trial_id])
        return command

    def _schedule_next(
        self,
        current_node: str,
        result: StageResult,
        current_trial_id: str | None,
        controller_mode: bool = False,
    ) -> None:
        payload_trial_id = result.payload.get("trial_id")
        if isinstance(payload_trial_id, str) and payload_trial_id:
            current_trial_id = payload_trial_id
        next_node = result.next_node if result.next_node else self.graph.route(current_node, result.status)
        if next_node is None:
            if self._has_inflight_trial_work():
                self.state.append_event(
                    self.study_id,
                    "study_completion_deferred",
                    {"current_node": current_node, "status": result.status},
                )
                return
            self.state.update_study_status(self.study_id, "completed")
            self.state.append_event(
                self.study_id,
                "study_completed",
                {"current_node": current_node, "status": result.status},
            )
            return
        next_cfg = self.graph.nodes[next_node]
        next_run_id = f"run_{uuid.uuid4().hex}"
        attempt = 0
        attempt_id = "attempt_000000"
        if current_trial_id:
            trial = self.state.get_trial(current_trial_id)
            attempt = trial.attempt_count + 1 if next_node == "run_trial" else trial.attempt_count
            attempt_id = f"attempt_{attempt:06d}"
        self.state.create_node_run(
            run_id=next_run_id,
            study_id=self.study_id,
            node_name=next_node,
            trial_id=current_trial_id,
            attempt=attempt,
        )
        self.state.append_event(
            self.study_id,
            "node_scheduled",
            {"run_id": next_run_id, "node_name": next_node, "trial_id": current_trial_id},
        )
        if next_cfg.executor == "local" and controller_mode:
            self.run_node(
                node_name=next_node,
                run_id=next_run_id,
                trial_id=current_trial_id,
                attempt_id=attempt_id,
                controller_mode=True,
            )

    def should_stop(self) -> bool:
        budget = self.config.autoresearch.budget
        stop_file = self.study_dir / "STOP"
        if stop_file.exists():
            return True
        max_wallclock_hours = float(getattr(budget, "max_wallclock_hours", 0) or 0)
        if max_wallclock_hours > 0:
            study = self.state.get_study(self.study_id)
            created_at = datetime.fromisoformat(study.created_at)
            now = datetime.now(timezone.utc)
            elapsed_hours = (now - created_at).total_seconds() / 3600.0
            if elapsed_hours >= max_wallclock_hours:
                return True
        max_gpu_hours = float(getattr(budget, "max_gpu_hours", 0) or 0)
        if max_gpu_hours > 0:
            used_gpu_hours = self.state.compute_trial_gpu_hours(self.study_id)
            if used_gpu_hours >= max_gpu_hours:
                return True
        trials = self.state.list_trials(self.study_id)
        if len(trials) >= int(getattr(budget, "max_trials", 0) or 0) > 0:
            return True
        parent_ids_with_retry = {
            trial.parent_trial_id
            for trial in trials
            if trial.parent_trial_id is not None
        }
        failures = sum(
            1
            for trial in trials
            if trial.status in {"failed", "timeout", "cancelled"}
            and trial.trial_id not in parent_ids_with_retry
        )
        if failures >= int(getattr(budget, "max_failures", 0) or 0) > 0:
            return True
        no_improve = int(getattr(budget, "no_improve_stop", 0) or 0)
        if no_improve > 0:
            streak = 0
            for trial in reversed(trials):
                if trial.status not in {"rejected", "accepted"}:
                    continue
                if trial.status == "accepted":
                    break
                streak += 1
            if streak >= no_improve and trials:
                return True
        return False

    def _has_inflight_trial_work(self) -> bool:
        active_trial_jobs = any(
            job.job_type == "trial" for job in self.state.list_active_jobs(self.study_id)
        )
        if active_trial_jobs:
            return True
        return any(
            node_run["trial_id"] is not None
            and node_run["status"] in {"pending", "running"}
            for node_run in self.state.list_node_runs(self.study_id)
        )

    def _refresh_active_jobs(
        self,
        trial_id: str | None = None,
        include_trial_jobs: bool = True,
    ) -> None:
        for job in self.state.list_active_jobs(self.study_id):
            if trial_id is not None and job.trial_id != trial_id:
                continue
            if not include_trial_jobs and job.job_type == "trial":
                continue
            handle = type("Handle", (), {})()
            handle.backend = job.backend
            handle.external_id = job.external_id
            handle.metadata = loads_json(job.command_json).get("metadata", {})
            executor = self.executor_local if job.backend == "local" else self.executor_trial
            status = executor.status(handle)
            self.state.update_job_status(job.job_id, status, external_id=job.external_id)
            if status in {"completed", "failed", "cancelled", "timeout"}:
                self._reconcile_terminal_job(job, status)

    def _reconcile_terminal_job(self, job: JobRecord, status: str) -> None:
        if not job.node_run_id:
            return
        node_run = self.state.get_node_run(job.node_run_id)
        if not node_run or node_run["status"] not in {"pending", "running"}:
            return

        if job.trial_id:
            trial = self.state.get_trial(job.trial_id)
            if trial and trial.status == "early_stopped":
                result = StageResult(
                    status="early_stopped",
                    message="Trial early stopped by keypoint agent",
                    payload={"trial_id": job.trial_id},
                )
            elif status == "completed":
                result = StageResult(status="success", message="External job completed")
            elif status == "timeout":
                trial.status = "timeout"
                self.state.update_trial(trial)
                result = StageResult(status="timeout", message="External job timed out")
            else:
                trial.status = "failed"
                self.state.update_trial(trial)
                result = StageResult(status="failure", message=f"External job ended with {status}")
        else:
            if status == "completed":
                result = StageResult(status="success", message="External job completed")
            elif status == "timeout":
                result = StageResult(status="timeout", message="External job timed out")
            else:
                result = StageResult(status="failure", message=f"External job ended with {status}")

        self.state.finish_node_run(job.node_run_id, result)
        self.state.append_event(
            self.study_id,
            "node_finished",
            {
                "run_id": job.node_run_id,
                "node_name": node_run["node_name"],
                "trial_id": node_run["trial_id"],
                "status": result.status,
                "message": result.message,
            },
        )
        self._schedule_next(
            current_node=node_run["node_name"],
            result=result,
            current_trial_id=node_run["trial_id"],
            controller_mode=True,
        )

    def _advance_local_ready_nodes(self, trial_id: str | None = None) -> None:
        for node_run in self.state.get_pending_node_runs(self.study_id):
            if trial_id is None:
                if node_run["trial_id"] is not None:
                    continue
            elif node_run["trial_id"] != trial_id:
                continue
            node_cfg = self.graph.nodes[node_run["node_name"]]
            if node_cfg.executor != "local":
                continue
            self.run_node(
                node_name=node_run["node_name"],
                run_id=node_run["run_id"],
                trial_id=node_run["trial_id"],
                attempt_id=f"attempt_{int(node_run['attempt']):06d}",
                controller_mode=True,
            )
            self._submit_pending_trial_nodes()

    def _schedule_trial_monitor(self, trial_id: str) -> None:
        active_monitors = [
            job
            for job in self.state.list_active_jobs(self.study_id)
            if job.job_type == "trial_monitor" and job.trial_id == trial_id
        ]
        if active_monitors:
            return

        monitor_cfg = getattr(self.config.autoresearch.executor, "trial_monitor", None)
        monitor_res_cfg = getattr(monitor_cfg, "resources", None) if monitor_cfg is not None else None
        if monitor_res_cfg is not None:
            tick_resources = Resources.from_config(monitor_res_cfg)
        else:
            tick_cfg = getattr(self.config.autoresearch.executor, "tick", None)
            tick_resources = Resources.from_config(
                getattr(tick_cfg, "resources", None) if tick_cfg is not None else None
            )
        interval = int(
            getattr(self.config.autoresearch, "tick_interval_sec_trial_active", 600) or 600
        )
        tick_dir = self.study_dir / "ticks" / f"trial_monitor_{uuid.uuid4().hex}"
        tick_dir.mkdir(parents=True, exist_ok=True)
        stdout = tick_dir / "stdout.log"
        stderr = tick_dir / "stderr.log"
        submit_stdout = tick_dir / "submit_stdout.log"
        submit_stderr = tick_dir / "submit_stderr.log"
        job_id = f"ar_trial_monitor_{uuid.uuid4().hex}"
        script_text = build_tick_loop_script(
            study_dir=self.study_dir,
            recipe_dir=self.recipe_dir,
            interval=interval,
            backend="slurm",
            slurm_time=tick_resources.time,
            slurm_partition=tick_resources.partition,
            slurm_account=tick_resources.account,
            slurm_mem=tick_resources.mem,
            slurm_qos=tick_resources.qos,
            slurm_constraint=tick_resources.constraint,
            slurm_reservation=tick_resources.reservation,
            slurm_nodelist=tick_resources.nodelist,
            trial_id=trial_id,
        )
        metadata = {
            "job_id": job_id,
            "command": ["bash", "-c", script_text],
            "resources": tick_resources.__dict__,
            "workdir": str(self.recipe_dir),
            "script_path": str(tick_dir / "job.sh"),
            "tick_dir": str(tick_dir),
            "submit_stdout_path": str(submit_stdout),
            "submit_stderr_path": str(submit_stderr),
            "job_type": "trial_monitor",
            "trial_id": trial_id,
        }
        spec = JobSpec(
            name=f"ar-{self.study_dir.name.replace('_', '-')}-{trial_id.replace('_', '-')}",
            command=["bash", "-c", script_text],
            workdir=self.recipe_dir,
            stdout=stdout,
            stderr=stderr,
            resources=tick_resources,
            env={
                "ESPNET_AR_STUDY_DIR": str(self.study_dir),
                "ESPNET_AR_TICK_JOB_ID": job_id,
            },
            metadata=metadata,
        )
        handle = SlurmExecutor(SlurmConfig()).submit(spec)
        self.state.create_tick_job(
            study_id=self.study_id,
            handle=handle,
            stdout_path=str(stdout),
            stderr_path=str(stderr),
        )

    @staticmethod
    def _parse_training_progress(
        metrics_csv_path: Path, columns: tuple[str, ...]
    ) -> int | None:
        """Return the latest populated integer progress value from metrics.csv."""
        try:
            with open(metrics_csv_path, newline="", encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
            # Lightning can emit rows with only train or validation fields.  Search
            # backwards rather than assuming the last row has every progress field.
            for row in reversed(rows):
                for col in columns:
                    val = row.get(col)
                    if val not in ("", None):
                        return int(float(val))
        except Exception:
            pass
        return None

    def _parse_training_epoch(self, metrics_csv_path: Path) -> int | None:
        """Return the latest epoch integer from a Lightning metrics CSV."""
        return self._parse_training_progress(metrics_csv_path, ("epoch", "Epoch"))

    def _parse_training_iteration(self, metrics_csv_path: Path) -> int | None:
        """Return the latest optimizer iteration from a Lightning metrics CSV."""
        return self._parse_training_progress(
            metrics_csv_path, ("step", "global_step", "Step", "GlobalStep")
        )

    @staticmethod
    def _elapsed_trial_seconds(started_at: str) -> int | None:
        """Return elapsed seconds since the trial worker entered the running state."""
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            return max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _load_checked_keypoints(path: Path) -> dict[str, list[int]]:
        """Load keypoint history, accepting the pre-trigger epoch-only format."""
        empty = {"epoch": [], "iteration": [], "elapsed_time": []}
        if not path.exists():
            return empty
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return empty
        if isinstance(data, list):
            # Version 1 recorded only epoch thresholds as a plain JSON array.
            empty["epoch"] = [int(value) for value in data]
            return empty
        if not isinstance(data, dict):
            return empty
        for kind in empty:
            values = data.get(kind, [])
            if isinstance(values, list):
                empty[kind] = [int(value) for value in values]
        return empty

    def _collect_keypoint_comparison(self, trial_id: str) -> str:
        """Gather CSV tails from other trials for comparison context."""
        trials_dir = self.study_dir / "trials"
        parts: list[str] = []
        for other_dir in sorted(trials_dir.iterdir()):
            if not other_dir.is_dir() or other_dir.name == trial_id:
                continue
            candidates = sorted(
                other_dir.glob("exp/csv_logs/**/metrics.csv"),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                continue
            try:
                with open(candidates[-1], newline="", encoding="utf-8", errors="replace") as f:
                    rows = list(csv.reader(f))
                if not rows:
                    continue
                header = ",".join(rows[0])
                tail = "\n".join(",".join(r) for r in rows[-5:])
                parts.append(f"{other_dir.name}:\n{header}\n{tail}")
            except Exception:
                continue
        return "\n\n".join(parts) if parts else "No other trial metrics available."

    def _call_keypoint_agent(
        self,
        trial,
        prompt: str,
        artifact_dir: Path,
    ) -> dict | None:
        """Call codex resume with keypoint check prompt, return parsed JSON or None."""
        import subprocess

        agent_cfg = self.config.autoresearch.agent
        base_cmd = [str(p) for p in list(getattr(agent_cfg, "command", []) or [])]
        if not base_cmd:
            return None
        with self.agent.session_lock() as shared_session_id:
            session_id = shared_session_id or trial.codex_thread_id
            if not session_id:
                self.logger.warning(
                    f"Keypoint agent skipped for {trial.trial_id}: no study agent session."
                )
                return None
            cmd = base_cmd + ["resume", session_id, "-"]

            codex_home = self.recipe_dir / ".codex"
            home_dir = codex_home / ".home"
            env = {
                **os.environ,
                "CODEX_HOME": str(codex_home),
                "HOME": str(home_dir),
                "XDG_CONFIG_HOME": str(codex_home / ".config"),
                "XDG_CACHE_HOME": str(codex_home / ".cache"),
            }

            prompt_path = artifact_dir / "agent_prompt.txt"
            stdout_path = artifact_dir / "agent_stdout.log"
            stderr_path = artifact_dir / "agent_stderr.log"
            prompt_path.write_text(prompt, encoding="utf-8")

            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=str(self.recipe_dir),
                env=env,
                check=False,
            )
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            self.logger.warning(
                f"Keypoint agent failed for {trial.trial_id} (rc={proc.returncode}): "
                f"{proc.stderr[:200]}"
            )
            return None

        for line in proc.stdout.splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "item.completed":
                text = (obj.get("item") or {}).get("text", "")
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        self.logger.warning(f"Keypoint agent output for {trial.trial_id} had no parseable JSON.")
        return None

    def _early_stop_trial(
        self,
        trial,
        trial_dir: Path,
        decision: dict,
    ) -> None:
        """Cancel GPU job, write EARLY_STOPPED.md, update trial status."""
        reason = str(decision.get("reason", "No reason provided"))
        key_metrics = decision.get("key_metrics", {})
        marker = (
            f"# Early Stopped\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Key Metrics:**\n"
            f"```json\n{json.dumps(key_metrics, ensure_ascii=False, indent=2)}\n```\n"
        )
        (trial_dir / "EARLY_STOPPED.md").write_text(marker, encoding="utf-8")
        trial.status = "early_stopped"
        self.state.update_trial(trial)
        self.logger.info(f"Early stopping {trial.trial_id}: {reason[:120]}")

        if os.environ.get("ESPNET_AR_GPU_SELF_CONTROLLER") == "1":
            # The GPU worker's parent supervisor observes this state and stops
            # its training child without cancelling the controller itself.
            (trial_dir / "EARLY_STOP_REQUESTED").write_text(reason, encoding="utf-8")
            return

        for job in self.state.list_active_jobs(self.study_id):
            if job.trial_id == trial.trial_id and job.job_type == "trial":
                handle = type("_H", (), {
                    "backend": job.backend,
                    "external_id": job.external_id,
                    "metadata": {},
                })()
                self.executor_trial.cancel(handle)
                self.state.update_job_status(job.job_id, "cancelled", external_id=job.external_id)

    def _maybe_check_keypoints(self, trial_id: str) -> None:
        """Ask the agent at configured epoch, iteration, or elapsed-time keypoints."""
        # Respect config flag
        early_stop_cfg = getattr(self.config.autoresearch, "early_stopping", None)
        if early_stop_cfg is not None and not bool(getattr(early_stop_cfg, "enabled", True)):
            return

        trial = self.state.get_trial(trial_id)
        if trial.status != "running":
            return
        if not getattr(trial, "codex_thread_id", None):
            return

        trial_dir = self.study_dir / "trials" / trial_id

        # Epoch keypoints preserve the original default: ten checks over a run.
        train_cfg_path = trial_dir / "resolved_training_config.yaml"
        total_epochs = 0
        if train_cfg_path.exists():
            try:
                import yaml

                with open(train_cfg_path, encoding="utf-8") as f:
                    train_cfg = yaml.safe_load(f) or {}
                total_epochs = int((train_cfg.get("trainer") or {}).get("max_epochs") or 0)
            except Exception:
                pass

        epoch_interval_value = getattr(early_stop_cfg, "epoch_interval", None)
        if epoch_interval_value is None:
            epoch_interval = max(1, round(total_epochs / 10)) if total_epochs > 0 else 0
        else:
            epoch_interval = max(0, int(epoch_interval_value or 0))
        iteration_interval = max(
            0, int(getattr(early_stop_cfg, "iteration_interval", 0) or 0)
        )
        elapsed_interval = max(
            0, int(getattr(early_stop_cfg, "elapsed_time_interval_sec", 0) or 0)
        )
        if not any((epoch_interval, iteration_interval, elapsed_interval)):
            return

        # Find metrics CSV
        candidates = sorted(
            trial_dir.glob("exp/csv_logs/**/metrics.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            return
        metrics_csv = candidates[-1]

        current_epoch = self._parse_training_epoch(metrics_csv)
        current_iteration = self._parse_training_iteration(metrics_csv)
        elapsed_seconds = self._elapsed_trial_seconds(trial.updated_at)

        due_keypoints: list[tuple[str, int, str]] = []
        if epoch_interval and current_epoch is not None and current_epoch >= epoch_interval:
            target = (current_epoch // epoch_interval) * epoch_interval
            due_keypoints.append(("epoch", target, f"epoch {target}/{total_epochs}"))
        if (
            iteration_interval
            and current_iteration is not None
            and current_iteration >= iteration_interval
        ):
            target = (current_iteration // iteration_interval) * iteration_interval
            due_keypoints.append(("iteration", target, f"iteration {target}"))
        if elapsed_interval and elapsed_seconds is not None and elapsed_seconds >= elapsed_interval:
            target = (elapsed_seconds // elapsed_interval) * elapsed_interval
            due_keypoints.append(("elapsed_time", target, f"elapsed GPU runtime {target}s"))
        if not due_keypoints:
            return

        kp_file = trial_dir / "keypoints_checked.json"
        with named_lock(trial_dir, "keypoints"):
            checked = self._load_checked_keypoints(kp_file)
            unchecked = [
                keypoint
                for keypoint in due_keypoints
                if keypoint[1] not in checked[keypoint[0]]
            ]
            if not unchecked:
                return

            # Read CSV tail
            try:
                with open(metrics_csv, newline="", encoding="utf-8", errors="replace") as f:
                    rows = list(csv.reader(f))
                if rows:
                    header = ",".join(rows[0])
                    tail_rows = "\n".join(",".join(r) for r in rows[-10:])
                    metrics_snapshot = f"{header}\n{tail_rows}"
                else:
                    metrics_snapshot = ""
            except Exception:
                metrics_snapshot = ""

            comparison = self._collect_keypoint_comparison(trial_id)

            progress = [
                f"epoch={current_epoch}/{total_epochs}" if current_epoch is not None else "epoch=unknown",
                f"iteration={current_iteration}" if current_iteration is not None else "iteration=unknown",
                f"elapsed_gpu_runtime_sec={elapsed_seconds}"
                if elapsed_seconds is not None
                else "elapsed_gpu_runtime_sec=unknown",
            ]
            prompt = "\n".join([
                f"Keypoint check for {trial_id}: {', '.join(label for _, _, label in unchecked)}.",
                f"Current progress: {', '.join(progress)}.",
                "",
                "Return exactly one JSON object with no markdown fences:",
                '{"should_stop": bool, "reason": "...", "key_metrics": {...}}',
                "",
                "Current training metrics (metrics.csv tail):",
                metrics_snapshot,
                "",
                "Other trials metrics for comparison:",
                comparison,
                "",
                "Should this trial be early stopped?",
                "Stop ONLY if the trajectory is clearly inferior to other trials and unlikely to recover.",
                "Be conservative — prefer to continue unless the evidence is strong.",
            ])

            primary_kind, primary_target, _ = unchecked[0]
            kp_dir = trial_dir / "keypoints" / f"{primary_kind}_{primary_target:06d}"
            kp_dir.mkdir(parents=True, exist_ok=True)
            (kp_dir / "triggers.json").write_text(
                dumps_json(
                    {
                        "triggers": [
                            {"kind": kind, "target": target, "label": label}
                            for kind, target, label in unchecked
                        ],
                        "current_epoch": current_epoch,
                        "current_iteration": current_iteration,
                        "elapsed_gpu_runtime_sec": elapsed_seconds,
                    }
                ),
                encoding="utf-8",
            )

            self.logger.info(
                f"Keypoint check for {trial_id}: "
                f"{', '.join(label for _, _, label in unchecked)}"
            )
            result = self._call_keypoint_agent(trial, prompt, kp_dir)

            for kind, target, _ in unchecked:
                checked[kind].append(target)
            kp_file.write_text(dumps_json(checked), encoding="utf-8")

        if result and result.get("should_stop"):
            self._early_stop_trial(trial, trial_dir, result)

    def _maybe_schedule_parallel_proposals(self) -> None:
        if "propose_trial" not in self.graph.nodes:
            return
        if self.bootstrap_node_name() != "propose_trial":
            return
        if self.should_stop():
            return
        budget = self.config.autoresearch.budget
        max_parallel = int(getattr(budget, "max_parallel_trials", 1) or 1)
        max_trials = int(getattr(budget, "max_trials", 0) or 0)
        with named_lock(self.study_dir, "proposal_scheduler"):
            trials = self.state.list_trials(self.study_id)
            node_runs = self.state.list_node_runs(self.study_id)
            active_trial_jobs = sum(
                1 for job in self.state.list_active_jobs(self.study_id) if job.job_type == "trial"
            )
            queued_run_trials = sum(
                1
                for node_run in node_runs
                if node_run["node_name"] == "run_trial"
                and node_run["status"] == "pending"
                and self.state.find_job_by_node_run(node_run["run_id"]) is None
            )
            inflight_proposals = sum(
                1
                for node_run in node_runs
                if node_run["node_name"] == "propose_trial"
                and node_run["status"] in {"pending", "running"}
            )
            inflight_debug_trials = sum(
                1
                for node_run in node_runs
                if node_run["node_name"] == "debug_trial"
                and node_run["status"] in {"pending", "running"}
            )
            occupied_slots = active_trial_jobs + queued_run_trials + inflight_proposals + inflight_debug_trials
            open_slots = max_parallel - occupied_slots
            if open_slots <= 0:
                return
            remaining_budget = open_slots
            if max_trials > 0:
                reserved_trials = len(trials) + inflight_proposals
                remaining_budget = min(remaining_budget, max_trials - reserved_trials)
            for _ in range(max(0, remaining_budget)):
                run_id = f"run_{uuid.uuid4().hex}"
                self.state.create_node_run(
                    run_id=run_id,
                    study_id=self.study_id,
                    node_name="propose_trial",
                )
                self.state.append_event(
                    self.study_id,
                    "node_scheduled",
                    {"run_id": run_id, "node_name": "propose_trial", "trial_id": None},
                )

    def _submit_pending_trial_nodes(self) -> None:
        """Submit pending trial nodes under one study-wide scheduler lock."""
        with named_lock(self.study_dir, "trial_submit"):
            self._submit_pending_trial_nodes_locked()

    def _submit_pending_trial_nodes_locked(self) -> None:
        budget = self.config.autoresearch.budget
        max_parallel = int(getattr(budget, "max_parallel_trials", 1) or 1)
        active_trial_jobs = sum(
            1
            for job in self.state.list_active_jobs(self.study_id)
            if job.node_run_id and self.state.get_node_run(job.node_run_id)["node_name"] == "run_trial"
        )
        for node_run in self.state.get_pending_node_runs(self.study_id):
            node_cfg = self.graph.nodes[node_run["node_name"]]
            if node_cfg.executor == "local":
                continue
            if active_trial_jobs >= max_parallel:
                return
            if self.state.find_job_by_node_run(node_run["run_id"]) is not None:
                continue
            trial_id = node_run["trial_id"]
            attempt = int(node_run["attempt"] or 0)
            attempt_id = f"attempt_{attempt:06d}"
            workdir = self.recipe_dir
            marker_dir = self.study_dir / "node_runs" / node_run["run_id"]
            if trial_id:
                attempt_dir = (
                    self.study_dir
                    / "trials"
                    / trial_id
                    / "attempts"
                    / attempt_id
                )
            else:
                attempt_dir = marker_dir
            stdout = attempt_dir / "stdout.log"
            stderr = attempt_dir / "stderr.log"
            command = self._build_run_node_command(
                node_name=node_run["node_name"],
                run_id=node_run["run_id"],
                trial_id=trial_id,
                attempt_id=attempt_id,
            )
            metadata = {
                "marker_dir": str(marker_dir),
                "script_path": str(attempt_dir / "job.sh"),
            }
            resources = self.trial_resources_default.merged_with(node_cfg.resources)
            if self.gpu_self_controller_enabled():
                controller_cfg = getattr(self.config.autoresearch, "gpu_self_controller", None)
                resources.cpus += max(0, int(getattr(controller_cfg, "monitor_cpus", 0) or 0))
            spec = JobSpec(
                name=self._job_name(
                    node_name=node_run["node_name"],
                    trial_id=trial_id,
                    attempt_id=attempt_id,
                ),
                command=command,
                workdir=workdir,
                stdout=stdout,
                stderr=stderr,
                resources=resources,
                env={
                    "ESPNET_AR_STUDY_DIR": str(self.study_dir),
                    "ESPNET_AR_TRIAL_ID": trial_id or "",
                    "ESPNET_AR_ATTEMPT_ID": attempt_id,
                    "ESPNET_AR_NODE_RUN_ID": node_run["run_id"],
                    "ESPNET_AR_GPU_SELF_CONTROLLER": "1"
                    if self.gpu_self_controller_enabled()
                    else "0",
                },
                metadata=metadata,
            )
            handle = self.executor_trial.submit(spec)
            self.state.create_job(
                JobRecord(
                    job_id=f"job_{uuid.uuid4().hex}",
                    study_id=self.study_id,
                    trial_id=trial_id,
                    node_run_id=node_run["run_id"],
                    backend=handle.backend,
                    external_id=handle.external_id,
                    status="submitted",
                    command_json=dumps_json(
                        {"command": command, "metadata": metadata},
                        indent=None,
                    ),
                    resources_json=dumps_json(spec.resources.__dict__, indent=None),
                    workdir=str(workdir),
                    stdout_path=str(stdout),
                    stderr_path=str(stderr),
                    job_type="trial",
                    created_at=utc_now(),
                    updated_at=utc_now(),
                )
            )
            active_trial_jobs += 1
            self.state.append_event(
                self.study_id,
                "job_submitted",
                {
                    "run_id": node_run["run_id"],
                    "node_name": node_run["node_name"],
                    "trial_id": trial_id,
                    "backend": handle.backend,
                    "external_id": handle.external_id,
                    "command": " ".join(shlex.quote(part) for part in command),
                },
            )
            if trial_id and not self.gpu_self_controller_enabled():
                self._schedule_trial_monitor(trial_id)

    async def run_forever(self, poll_interval: float = 30.0) -> None:
        while True:
            self.tick()
            if self.should_stop():
                pending = self.state.get_pending_node_runs(self.study_id)
                active = self.state.list_active_jobs(self.study_id)
                if not pending and not active:
                    return
            await asyncio.sleep(poll_interval)

    def tick(self, trial_id: str | None = None) -> None:
        if trial_id is None and not self.state.list_node_runs(self.study_id):
            self.state.create_node_run(
                run_id=f"run_{uuid.uuid4().hex}",
                study_id=self.study_id,
                node_name=self.bootstrap_node_name(),
            )
        self._refresh_active_jobs(
            trial_id=trial_id,
            include_trial_jobs=trial_id is not None,
        )
        if trial_id is not None:
            try:
                self._maybe_check_keypoints(trial_id)
            except Exception as exc:
                self.logger.warning(f"Keypoint check failed for {trial_id}: {exc}")
        if trial_id is None:
            self._maybe_schedule_parallel_proposals()
        self._submit_pending_trial_nodes()
        self._advance_local_ready_nodes(trial_id=trial_id)
        self._submit_pending_trial_nodes()
