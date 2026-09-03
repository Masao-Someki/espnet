"""Run one trial end-to-end."""

from __future__ import annotations

import os
import shlex
import subprocess
import shutil
import time
from pathlib import Path

from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.serialization import save_yaml
from espnet3.autoresearch.core.stage import AutoResearchStage


class RunTrialStage(AutoResearchStage):
    """Execute training, inference, and measurement for one trial."""

    def _copy_training_curve_csv(self, trial_dir: Path, attempt_dir: Path) -> None:
        csv_candidates = sorted(
            (trial_dir / "exp").glob("csv_logs/**/metrics.csv"),
            key=lambda path: path.stat().st_mtime,
        )
        if not csv_candidates:
            return
        latest = csv_candidates[-1]
        shutil.copyfile(latest, trial_dir / "training_metrics.csv")
        shutil.copyfile(latest, attempt_dir / "training_metrics.csv")

    def _run_cmd(self, cmd: list[str], cwd: Path, log_path: Path, timeout: int | None):
        with open(log_path, "a", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        return proc

    @staticmethod
    def _is_training_command(command: list[str]) -> bool:
        try:
            return command[command.index("--stages") + 1] == "train"
        except (ValueError, IndexError):
            # A custom command template represents the complete trial workload.
            return True

    def _run_with_gpu_monitor(self, context, command, log_path, env, timeout):
        """Run training while the GPU worker performs cooperative early-stop checks."""
        interval = context.scheduler.gpu_self_monitor_interval_sec()
        started = time.monotonic()
        next_check = started + interval
        with open(log_path, "a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                command,
                cwd=str(context.recipe_dir),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while proc.poll() is None:
                now = time.monotonic()
                if timeout is not None and now - started >= timeout:
                    proc.terminate()
                    proc.wait(timeout=30)
                    return "timeout", None
                if now >= next_check:
                    try:
                        context.scheduler._maybe_check_keypoints(
                            context.current_trial.trial_id
                        )
                    except Exception as exc:  # noqa: BLE001
                        context.logger.warning(f"In-job early-stop check failed: {exc}")
                    current = context.state.get_trial(context.current_trial.trial_id)
                    if current.status == "early_stopped":
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        return "early_stopped", None
                    next_check = now + interval
                time.sleep(min(5.0, max(0.1, next_check - time.monotonic())))
        return "completed", proc

    # Stages that only need --training_config
    _TRAIN_ONLY_STAGES = frozenset({"create_dataset", "train_tokenizer", "collect_stats", "train"})
    # Stages that also need --inference_config
    _INFER_STAGES = frozenset({"infer"})
    # Stages that need all three configs
    _MEASURE_STAGES = frozenset({"measure"})

    def _stage_command(
        self,
        stage: str,
        training_cfg: Path,
        inference_cfg: Path,
        metrics_cfg: Path,
    ) -> list[str]:
        cmd = ["python", "run.py", "--stages", stage, "--training_config", str(training_cfg)]
        if stage in self._INFER_STAGES:
            cmd += ["--inference_config", str(inference_cfg)]
        elif stage in self._MEASURE_STAGES:
            cmd += [
                "--inference_config", str(inference_cfg),
                "--metrics_config", str(metrics_cfg),
            ]
        return cmd

    def _build_default_commands(
        self,
        training_cfg: Path,
        inference_cfg: Path,
        metrics_cfg: Path,
        extra_stages: list[str] | None = None,
    ) -> list[list[str]]:
        cmds: list[list[str]] = []
        for stage in (extra_stages or []):
            cmds.append(self._stage_command(stage, training_cfg, inference_cfg, metrics_cfg))
        cmds.append(self._stage_command("train", training_cfg, inference_cfg, metrics_cfg))
        cmds.append(self._stage_command("infer", training_cfg, inference_cfg, metrics_cfg))
        cmds.append(self._stage_command("measure", training_cfg, inference_cfg, metrics_cfg))
        return cmds

    def _build_commands(self, context, trial_dir: Path, attempt_dir: Path) -> list[list[str]]:
        training_cfg = trial_dir / "resolved_training_config.yaml"
        inference_cfg = trial_dir / "resolved_inference_config.yaml"
        metrics_cfg = trial_dir / "resolved_metrics_config.yaml"
        template = list(
            getattr(context.config.autoresearch.trial, "command_template", []) or []
        )
        if template:
            values = {
                "trial_config": str(training_cfg),
                "training_config": str(training_cfg),
                "inference_config": str(inference_cfg),
                "metrics_config": str(metrics_cfg),
                "trial_dir": str(trial_dir),
                "attempt_dir": str(attempt_dir),
                "recipe_dir": str(context.recipe_dir),
            }
            return [[str(part).format(**values) for part in template]]

        trial = context.current_trial
        extra_stages = list(getattr(trial, "extra_stages", None) or [])
        return self._build_default_commands(training_cfg, inference_cfg, metrics_cfg, extra_stages)

    def run(self, context) -> StageResult:
        trial = context.current_trial
        assert trial is not None
        trial_dir = context.trial_dir()
        attempt_dir = trial_dir / "attempts" / context.attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        training_cfg = trial_dir / "resolved_training_config.yaml"
        inference_cfg = trial_dir / "resolved_inference_config.yaml"
        metrics_cfg = trial_dir / "resolved_metrics_config.yaml"
        train_log = trial_dir / "train.log"
        eval_log = trial_dir / "eval.log"
        commands = self._build_commands(context, trial_dir, attempt_dir)

        script_lines = [
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            f"cd {context.recipe_dir}",
            *[" ".join(shlex.quote(str(part)) for part in command) for command in commands],
            "",
        ]
        script_path = trial_dir / "train_eval.sh"
        script_path.write_text("\n".join(script_lines), encoding="utf-8")
        script_path.chmod(0o755)

        trial.status = "running"
        trial.attempt_count += 1
        context.state.update_trial(trial)
        timeout = getattr(context.config.autoresearch.trial, "timeout_sec", None)
        save_yaml(
            attempt_dir / "job.yaml",
            {
                "trial_id": trial.trial_id,
                "attempt_id": context.attempt_id,
                "commands": commands,
            },
        )
        env = {**os.environ, "ESPNET_AR_TRIAL_DIR": str(trial_dir)}
        try:
            proc = None
            for index, command in enumerate(commands):
                log_path = train_log if index == 0 else eval_log
                if (
                    os.environ.get("ESPNET_AR_GPU_SELF_CONTROLLER") == "1"
                    and self._is_training_command(command)
                ):
                    monitor_status, proc = self._run_with_gpu_monitor(
                        context, command, log_path, env, timeout
                    )
                    if monitor_status == "early_stopped":
                        return StageResult(status="early_stopped", message="Trial early stopped")
                    if monitor_status == "timeout":
                        trial.status = "timeout"
                        context.state.update_trial(trial)
                        return StageResult(status="timeout", message="Trial timed out")
                else:
                    with open(log_path, "a", encoding="utf-8") as handle:
                        proc = subprocess.run(
                            command,
                            cwd=str(context.recipe_dir),
                            env=env,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=timeout,
                            check=False,
                        )
                if proc.returncode != 0:
                    trial.status = "failed"
                    context.state.update_trial(trial)
                    return StageResult(
                        status="failure",
                        message=f"Command failed: {' '.join(command)}",
                    )
        except subprocess.TimeoutExpired:
            trial.status = "timeout"
            context.state.update_trial(trial)
            return StageResult(status="timeout", message="Trial timed out")

        trial.status = "completed"
        context.state.update_trial(trial)
        self._copy_training_curve_csv(trial_dir, attempt_dir)
        metrics_path = trial_dir / "inference" / "metrics.json"
        if metrics_path.exists():
            target_metrics = trial_dir / "metrics.json"
            target_metrics.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")
        return StageResult(
            status="success",
            message="Trial completed",
            artifacts={
                "train_log": str(train_log),
                "eval_log": str(eval_log),
                "training_metrics_csv": str(trial_dir / "training_metrics.csv"),
                "metrics": str(trial_dir / "metrics.json"),
            },
        )
