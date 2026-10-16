"""Build study knowledge context."""

from __future__ import annotations

from pathlib import Path

from espnet3.cli.autoresearch.core.models import StageResult
from espnet3.cli.autoresearch.core.serialization import atomic_write_text
from espnet3.cli.autoresearch.core.stage import AutoResearchStage


class ResearchContextStage(AutoResearchStage):
    """Collect repository context and prior trial summaries."""

    def _load_text(self, path: Path) -> str:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def run(self, context) -> StageResult:
        ar_cfg = context.config.autoresearch
        knowledge_dir = context.study_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        repo_path = knowledge_dir / "repo_context.md"
        pack_path = knowledge_dir / "knowledge_pack.md"
        if repo_path.is_file() and pack_path.is_file():
            return StageResult(
                status="success",
                message="Knowledge context reused",
                artifacts={
                    "repo_context": str(repo_path),
                    "knowledge_pack": str(pack_path),
                },
            )
        program_text = self._load_text(context.study_dir / str(ar_cfg.objective_file))

        repo_lines = [
            "# Repo Context",
            "",
            f"- Recipe dir: {context.recipe_dir}",
            f"- Study dir: {context.study_dir}",
            f"- Training config: {getattr(ar_cfg.recipe, 'training_config', '')}",
            f"- Inference config: {getattr(ar_cfg.recipe, 'inference_config', '')}",
            f"- Metrics config: {getattr(ar_cfg.recipe, 'metrics_config', '')}",
            "",
            "## Objective",
            program_text.strip() or "(empty)",
            "",
        ]
        trials = context.state.list_trials(context.scheduler.study_id)
        repo_lines.append("## Prior Trials")
        if not trials:
            repo_lines.append("- none")
        else:
            for trial in trials[-10:]:
                repo_lines.append(
                    f"- {trial.trial_id}: status={trial.status}, score={trial.score}, "
                    f"decision={trial.decision or ''}"
                )
        repo_lines.append("")
        atomic_write_text(repo_path, "\n".join(repo_lines))

        atomic_write_text(
            pack_path,
            "\n\n".join(
                [
                    "# Knowledge Pack",
                    "## Objective",
                    program_text.strip() or "(empty)",
                    repo_path.read_text(encoding="utf-8"),
                ]
            ),
        )
        return StageResult(
            status="success",
            message="Knowledge context prepared",
            artifacts={
                "repo_context": str(repo_path),
                "knowledge_pack": str(pack_path),
            },
        )
