"""Reflect on the latest trial and decide whether to continue."""

from __future__ import annotations

from espnet3.autoresearch.agents.interface import AgentRequest
from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.stage import AutoResearchStage


class ReflectStage(AutoResearchStage):
    """Write a short reflection and stop/continue signal."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        trial = context.current_trial
        assert trial is not None

        # Early-stopped trials: use EARLY_STOPPED.md content, skip agent call
        if trial.status == "early_stopped":
            early_stopped_path = context.trial_dir() / "EARLY_STOPPED.md"
            if early_stopped_path.exists():
                reflection = early_stopped_path.read_text(encoding="utf-8").strip()
            else:
                reflection = f"Trial {trial.trial_id} was early stopped by the agent."
            next_suggestion = (
                "This config was early stopped due to poor training trajectory. "
                "Avoid repeating this configuration direction."
            )
            (context.trial_dir() / "reflection.md").write_text(reflection, encoding="utf-8")
            trial.reflection = reflection
            trial.next_suggestion = next_suggestion
            context.state.update_trial(trial)
            return StageResult(
                status="continue",
                message=f"Reflected on early-stopped trial {trial.trial_id}",
            )

        trials_csv = context.study_dir / "trials.csv"
        request = AgentRequest(
            task="reflect",
            objective=(context.study_dir / "program.md").read_text(encoding="utf-8"),
            knowledge=(context.study_dir / "knowledge" / "knowledge_pack.md").read_text(
                encoding="utf-8"
            )
            if (context.study_dir / "knowledge" / "knowledge_pack.md").exists()
            else "",
            repo_context=(context.study_dir / "knowledge" / "repo_context.md").read_text(
                encoding="utf-8"
            )
            if (context.study_dir / "knowledge" / "repo_context.md").exists()
            else "",
            trial_history_csv=trials_csv.read_text(encoding="utf-8")
            if trials_csv.exists()
            else "",
            latest_metrics=trial.metrics,
            latest_logs={},
            allowed_actions=["reflection_only"],
            output_schema={"reflection": "string", "next_suggestion": "string"},
        )
        artifact_dir = context.trial_dir()
        response = context.agent.run(request, artifact_dir=artifact_dir)
        if response.status == "success":
            structured = dict(response.structured or {})
            reflection = str(structured.get("reflection", response.content or ""))
            next_suggestion = str(structured.get("next_suggestion", ""))
        elif context.config.autoresearch.agent.type == "file":
            return StageResult(status="waiting_agent", message=response.message)
        else:
            reflection = (
                f"Trial {trial.trial_id} ended with status={trial.status}, "
                f"score={trial.score}."
            )
            next_suggestion = "Try the next configured candidate patch."
        (context.trial_dir() / "reflection.md").write_text(reflection, encoding="utf-8")
        trial.reflection = reflection
        trial.next_suggestion = next_suggestion
        context.state.update_trial(trial)
        if trial.status in {"failed", "timeout"}:
            return StageResult(
                status="continue",
                message="Continue search after failed trial reflection",
            )
        if context.scheduler.should_stop():
            return StageResult(status="done", message="Stopping after reflection")
        return StageResult(status="continue", message="Continue search")
