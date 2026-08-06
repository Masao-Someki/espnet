"""Build study knowledge context."""

from __future__ import annotations

from pathlib import Path

from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.serialization import atomic_write_text
from espnet3.autoresearch.core.stage import AutoResearchStage


class ResearchContextStage(AutoResearchStage):
    """Collect repo context, optional web notes, and trial summaries."""

    def _load_text(self, path: Path) -> str:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def run(self, context) -> StageResult:
        ar_cfg = context.config.autoresearch
        knowledge_dir = context.study_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        web_path = knowledge_dir / "web_research.md"
        repo_path = knowledge_dir / "repo_context.md"
        pack_path = knowledge_dir / "knowledge_pack.md"
        if web_path.is_file() and repo_path.is_file() and pack_path.is_file():
            return StageResult(
                status="success",
                message="Knowledge context reused",
                artifacts={
                    "web_research": str(web_path),
                    "repo_context": str(repo_path),
                    "knowledge_pack": str(pack_path),
                },
            )
        program_text = self._load_text(context.study_dir / str(ar_cfg.objective_file))

        query_texts = []
        for template in list(getattr(ar_cfg, "search_queries", []) or []):
            rendered = str(template).format(
                task=getattr(ar_cfg, "task", "ASR"),
                dataset=getattr(ar_cfg, "dataset", "mini_an4"),
                model=getattr(ar_cfg, "model", "transformer"),
                year=getattr(ar_cfg, "year", "2026"),
            )
            query_texts.append(rendered)

        web_lines = ["# Web Research", ""]
        for query in query_texts:
            results = context.search.search(query, max_results=5)
            web_lines.append(f"## {query}")
            if not results:
                web_lines.append("- No results")
                web_lines.append("")
                continue
            for item in results:
                web_lines.append(f"- {item.title} | {item.url}")
                if item.snippet:
                    web_lines.append(f"  {item.snippet}")
            web_lines.append("")
        atomic_write_text(web_path, "\n".join(web_lines))

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
                    web_path.read_text(encoding="utf-8"),
                ]
            ),
        )
        return StageResult(
            status="success",
            message="Knowledge context prepared",
            artifacts={
                "web_research": str(web_path),
                "repo_context": str(repo_path),
                "knowledge_pack": str(pack_path),
            },
        )
