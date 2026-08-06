"""Collect metrics for a completed trial."""

from __future__ import annotations

from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.stage import AutoResearchStage
from espnet3.autoresearch.observers.metrics import extract_metrics, resolve_primary_score


class CollectResultStage(AutoResearchStage):
    """Read metric files and update persistent trial state."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        trial = context.current_trial
        assert trial is not None
        metric_cfg = context.config.autoresearch.metric
        try:
            metrics = extract_metrics(
                context.trial_dir(),
                metric_cfg.extraction,
            )
            score = resolve_primary_score(metrics, str(metric_cfg.name))
        except Exception as exc:  # noqa: BLE001
            trial.status = "failed"
            context.state.update_trial(trial)
            return StageResult(status="failure", message=str(exc))

        trial.metrics = metrics
        trial.score = score
        trial.score_name = str(metric_cfg.name)
        if trial.status not in {"accepted", "rejected"}:
            trial.status = "completed"
        context.state.update_trial(trial)
        return StageResult(
            status="success",
            message=f"Collected score={score}",
            payload={"score": score},
        )
